#!/usr/bin/env node

import { Command } from "commander";
import * as fs from "fs";
import { spawnSync } from "child_process";
import chalk from "chalk";
import { parseFile } from "./ast-parser.js";
import {
  computeSimilarity,
  printSimilarityReport,
  combinedSimilarityWithContent,
  identifierCosineSimilarity,
  canonicalIdentifierCosineSimilarity,
  canonicalIdentifierJaccardSimilarity,
} from "./similarity.js";
import {
  parseDirectory,
  buildDepGraph,
  rankPortingPriority,
  findMissingFiles,
  printCodebaseStats,
} from "./codebase.js";
import { Language } from "./types.js";

const program = new Command();

function refusePipedStdio(): void {
  // Refuse to run when stdout or stderr is piped via a shell pipeline.
  // This blocks `ast_distance ... | sed/grep/...` which has caused model-driven
  // wrappers to silently filter or truncate dashboards.
  //
  // We intentionally allow non-terminal stdout when the caller captures output
  // directly (e.g. a wrapper process reading from a pipe), because that is not
  // the same failure mode as shell filtering commands like `sed`/`grep`.
  function detectPipelinePeerProcess(): boolean {
    const selfPid = process.pid;
    const ps = spawnSync(
      "ps",
      ["-A", "-o", "pid=", "-o", "ppid=", "-o", "pgid="],
      { encoding: "utf8" },
    );
    if (ps.status !== 0 || typeof ps.stdout !== "string") {
      return true;
    }

    type Row = { pid: number; ppid: number; pgid: number };
    const rows: Row[] = [];
    const ppidByPid = new Map<number, number>();

    let pgid: number | null = null;
    for (const line of ps.stdout.split("\n")) {
      const parts = line.trim().split(/\s+/);
      if (parts.length < 3) continue;
      const pid = Number(parts[0]);
      const ppid = Number(parts[1]);
      const pg = Number(parts[2]);
      if (!Number.isFinite(pid) || !Number.isFinite(ppid) || !Number.isFinite(pg)) {
        continue;
      }
      rows.push({ pid, ppid, pgid: pg });
      ppidByPid.set(pid, ppid);
      if (pid === selfPid) {
        pgid = pg;
      }
    }
    if (pgid === null) {
      return true;
    }

    const ancestors = new Set<number>();
    let cur = selfPid;
    for (let i = 0; i < 64; i++) {
      const parent = ppidByPid.get(cur) ?? 0;
      if (parent <= 0) break;
      if (ancestors.has(parent)) break;
      ancestors.add(parent);
      cur = parent;
    }

    for (const r of rows) {
      if (r.pgid !== pgid) continue;
      if (r.pid === selfPid) continue;
      if (ancestors.has(r.pid)) continue;
      if (r.ppid === selfPid) continue; // our own child
      return true;
    }

    return false;
  }

  try {
    const out = fs.fstatSync(process.stdout.fd);
    if (out.isFIFO()) {
      if (!detectPipelinePeerProcess()) {
        return;
      }
      process.stderr.write(
        "Error: stdout is piped to another program.\n" +
          "ast_distance does not support piping (|).\n" +
          "Run it directly in a terminal.\n",
      );
      process.exit(2);
    }
  } catch {
    // ignore
  }
  try {
    const err = fs.fstatSync(process.stderr.fd);
    if (err.isFIFO()) {
      if (!detectPipelinePeerProcess()) {
        return;
      }
      process.stdout.write(
        "Error: stderr is piped to another program.\n" +
          "ast_distance does not support piping (|).\n" +
          "Run it directly in a terminal.\n",
      );
      process.exit(2);
    }
  } catch {
    // ignore
  }
}

refusePipedStdio();

function parseLanguage(lang: string): Language {
  const normalized = lang.toLowerCase();
  switch (normalized) {
    case "typescript":
    case "ts":
    case "tsx":
      return Language.TYPESCRIPT;
    case "rust":
    case "rs":
      return Language.RUST;
    case "kotlin":
    case "kt":
      return Language.KOTLIN;
    case "cpp":
    case "c++":
      return Language.CPP;
    case "c":
      return Language.C;
    default:
      throw new Error(
        `Unknown language: ${lang} (use typescript, rust, kotlin, cpp, or c)`,
      );
  }
}

function dumpTree(node: any, indent: number = 0): void {
  const pad = "  ".repeat(indent);
  console.log(`${pad}${node.nodeType} (${node.label})`);

  for (const child of node.children) {
    dumpTree(child, indent + 1);
  }
}

function containsMacroRules(filePath: string): boolean {
  try {
    return fs.readFileSync(filePath, "utf-8").includes("macro_rules!");
  } catch {
    return false;
  }
}

program
  .name("ast-distance")
  .description("Cross-language AST comparison and porting analysis tool")
  .version("0.1.0");

// Compare two files
program
  .argument("<file1>")
  .argument("<lang1>")
  .argument("<file2>")
  .argument("<lang2>")
  .action(async (file1, lang1, file2, lang2) => {
    const l1 = parseLanguage(lang1);
    const l2 = parseLanguage(lang2);

    const result1 = parseFile(file1, l1);
    const result2 = parseFile(file2, l2);

    if (!result1 || !result2) {
      console.error(chalk.red("Failed to parse files"));
      process.exit(1);
    }

    const macroFriendly =
      (l1 === Language.RUST && containsMacroRules(file1)) ||
      (l2 === Language.RUST && containsMacroRules(file2));

    printSimilarityReport(result1, result2, {
      macroFriendly,
      includeContent: false,
    });

    const shape = computeSimilarity(result1, result2, { macroFriendly });
    const contentScore = combinedSimilarityWithContent(result1, result2, {
      macroFriendly,
    });

    console.log("=== Identifier Content Analysis ===");
    console.log(
      `Identifiers:          ${result1.identifierStats.totalIdentifiers} vs ${result2.identifierStats.totalIdentifiers}`,
    );
    console.log(
      `Unique (raw):         ${result1.identifierStats.identifierFreq.size} vs ${result2.identifierStats.identifierFreq.size}`,
    );
    console.log(
      `Unique (canonical):   ${result1.identifierStats.canonicalFreq.size} vs ${result2.identifierStats.canonicalFreq.size}`,
    );
    console.log(
      `Raw cosine:           ${identifierCosineSimilarity(result1.identifierStats, result2.identifierStats).toFixed(4)}`,
    );
    console.log(
      `Canonical cosine:     ${canonicalIdentifierCosineSimilarity(result1.identifierStats, result2.identifierStats).toFixed(4)}`,
    );
    console.log(
      `Canonical jaccard:    ${canonicalIdentifierJaccardSimilarity(result1.identifierStats, result2.identifierStats).toFixed(4)}`,
    );

    if (result1.hasStubBodies || result2.hasStubBodies) {
      console.log("\n*** STUB DETECTED ***");
      if (result1.hasStubBodies) {
        console.log(`  ${file1} has TODO/stub/placeholder in function bodies`);
      }
      if (result2.hasStubBodies) {
        console.log(`  ${file2} has TODO/stub/placeholder in function bodies`);
      }
      console.log("  Content-Aware Score forced to 0.0000");
    } else {
      console.log(`\nContent-Aware Score:  ${contentScore.toFixed(4)}`);
    }

    console.log(`Shape Combined Score: ${shape.combined.toFixed(4)}`);
    console.log(`Shape Histogram:      ${shape.cosineHistogram.toFixed(4)}`);
  });

// Dump AST
program
  .command("dump")
  .argument("<file>")
  .argument("<lang>")
  .action((file, lang) => {
    const result = parseFile(file, parseLanguage(lang));

    if (!result) {
      console.error(chalk.red("Failed to parse file"));
      process.exit(1);
    }

    console.log(`=== AST Dump: ${file} ===\n`);
    dumpTree(result.tree);
  });

// Scan directory
program
  .command("scan")
  .argument("<directory>")
  .argument("<lang>")
  .action((directory, lang) => {
    console.log(`Scanning ${directory} for ${lang} files...\n`);

    const files = parseDirectory(directory, parseLanguage(lang));
    printCodebaseStats(files);

    console.log("=== File List ===\n");

    for (const [relPath, file] of files.entries()) {
      const status = file.isStub ? chalk.yellow("[STUB]") : chalk.green("[OK]");
      const linesStr = file.lineCount.toString().padStart(5);
      console.log(`${status} ${linesStr} lines  ${relPath}`);
    }
  });

// Deep analysis
program
  .command("deep")
  .argument("<srcDir>")
  .argument("<srcLang>")
  .argument("<tgtDir>")
  .argument("<tgtLang>")
  .action(async (srcDir, srcLang, tgtDir, tgtLang) => {
    console.log(chalk.bold("=== Deep Codebase Analysis ===\n"));
    console.log(`Source: ${srcDir} (${srcLang})`);
    console.log(`Target: ${tgtDir} (${tgtLang})\n`);

    const srcFiles = parseDirectory(srcDir, parseLanguage(srcLang));
    const tgtFiles = parseDirectory(tgtDir, parseLanguage(tgtLang));

    console.log(chalk.bold("Source Statistics:"));
    printCodebaseStats(srcFiles);

    console.log(chalk.bold("Target Statistics:"));
    printCodebaseStats(tgtFiles);

    const depGraph = buildDepGraph(srcFiles);

    const missing = findMissingFiles(srcFiles, tgtFiles);
    console.log(chalk.bold(`\nMissing Files: ${missing.length}\n`));

    for (const item of missing.slice(0, 20)) {
      console.log(`  ${item.path}`);
    }

    if (missing.length > 20) {
      console.log(`  ... and ${missing.length - 20} more`);
    }

    const rankings = rankPortingPriority(srcFiles, tgtFiles, depGraph);
    console.log(chalk.bold("\nPorting Priorities (Top 10):\n"));

    for (const item of rankings.slice(0, 10)) {
      const status =
        item.dependents > 0
          ? chalk.yellow(item.dependents.toString())
          : chalk.gray("0");
      console.log(`  ${status} dependents  ${item.path}`);
    }
  });

// Missing files
program
  .command("missing")
  .argument("<srcDir>")
  .argument("<srcLang>")
  .argument("<tgtDir>")
  .argument("<tgtLang>")
  .action((srcDir, srcLang, tgtDir, tgtLang) => {
    const srcFiles = parseDirectory(srcDir, parseLanguage(srcLang));
    const tgtFiles = parseDirectory(tgtDir, parseLanguage(tgtLang));

    const missing = findMissingFiles(srcFiles, tgtFiles);

    console.log(chalk.bold(`Missing Files: ${missing.length}\n`));

    for (const item of missing) {
      console.log(`  ${item.path}`);
    }
  });

// Scan for TODOs
program
  .command("todos")
  .argument("<directory>")
  .action((directory) => {
    const files = parseDirectory(directory, Language.TYPESCRIPT);
    let totalTodos = 0;

    console.log(chalk.bold("=== TODO Scan ===\n"));

    for (const file of files.values()) {
      for (const filePath of file.paths) {
        const content = fs.readFileSync(filePath, "utf-8");
        const lines = content.split("\n");

        lines.forEach((line, idx) => {
          const matches = line.match(/TODO[:\s](.*)/i);
          if (matches) {
            totalTodos++;
            console.log(`${chalk.cyan(file.filename)}:${idx + 1}`);
            console.log(`  ${matches[1].trim()}\n`);
          }
        });
      }
    }

    console.log(`Total TODOs: ${totalTodos}`);
  });

program.parse();
