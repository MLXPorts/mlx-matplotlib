import { ParseResult, Language } from "./types.js";
import * as fs from "fs";
import * as path from "path";
import { parseFile } from "./ast-parser.js";

const SKIP_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  "target",
  "__pycache__",
  "_deps",
]);

const HEADER_EXTENSIONS = new Set([".hpp", ".h", ".hxx", ".hh"]);
const VARIANT_SUFFIXES = [
  ".common",
  ".concurrent",
  ".native",
  ".common_native",
  ".darwin",
  ".apple",
];

const LANGUAGE_EXTENSIONS: Record<string, string[]> = {
  [Language.TYPESCRIPT]: [".ts", ".tsx"],
  [Language.RUST]: [".rs"],
  [Language.KOTLIN]: [".kt", ".kts"],
  [Language.CPP]: [".cpp", ".cc", ".cxx", ".hpp", ".h", ".hxx", ".hh"],
  [Language.C]: [".c", ".h"],
};

/**
 * File entry in a codebase (logical unit).
 */
export interface FileEntry {
  filename: string;
  filepath: string;
  relativePath: string;
  language: Language;
  paths: string[];
  stem: string;
  qualifiedName: string;
  parseResult: ParseResult | null;
  imports: string[];
  exports: string[];
  lineCount: number;
  isStub: boolean;
}

/**
 * Dependency graph node.
 */
export interface DepNode {
  relativePath: string;
  dependents: string[];
  dependencies: string[];
}

/**
 * Dependency graph for a codebase.
 */
export interface DepGraph {
  nodes: Map<string, DepNode>;
}

function normalizeRelPath(relPath: string): string {
  return relPath.replace(/\\/g, "/");
}

function normalizeStem(stem: string): string {
  for (const suffix of VARIANT_SUFFIXES) {
    if (stem.endsWith(suffix)) {
      return stem.slice(0, -suffix.length);
    }
  }
  return stem;
}

function normalizeName(name: string): string {
  return name.replace(/_/g, "").toLowerCase();
}

function makeQualifiedName(relPath: string): string {
  const p = path.parse(relPath);
  const parentParts = normalizeRelPath(p.dir)
    .split("/")
    .filter((part) => part.length > 0 && part !== "." && part !== "src");

  if (parentParts.length > 0) {
    return `${parentParts[parentParts.length - 1]}.${p.name}`;
  }
  return p.name;
}

function shouldParseFile(filePath: string, language: Language): boolean {
  const ext = path.extname(filePath);
  return (LANGUAGE_EXTENSIONS[language] || []).includes(ext);
}

function collectFiles(rootDir: string): string[] {
  const out: string[] = [];

  function walk(currentDir: string) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) {
          continue;
        }
        walk(fullPath);
        continue;
      }

      if (entry.isFile()) {
        out.push(fullPath);
      }
    }
  }

  walk(rootDir);
  return out;
}

function parseLogicalEntry(file: FileEntry): void {
  file.paths.sort((a, b) => {
    const aIsHeader = HEADER_EXTENSIONS.has(path.extname(a));
    const bIsHeader = HEADER_EXTENSIONS.has(path.extname(b));
    if (aIsHeader !== bIsHeader) return aIsHeader ? -1 : 1;
    return a.length - b.length;
  });

  file.filepath = file.paths[0] || file.filepath;
  file.parseResult = parseFile(file.paths, file.language);
  file.imports = file.parseResult?.importPaths || [];
  file.exports = file.parseResult?.exportPaths || [];

  let lineCount = 0;
  let anyPartStub = false;
  let allPartStub = file.paths.length > 0;

  for (const filePath of file.paths) {
    const content = fs.readFileSync(filePath, "utf-8");
    lineCount += content.split("\n").length;

    const partStub =
      /TODO|FIXME|placeholder|not implemented|not yet implemented/i.test(content);
    anyPartStub = anyPartStub || partStub;
    allPartStub = allPartStub && partStub;
  }

  file.lineCount = lineCount;

  // Keep the same rule as C++/Python logical grouping:
  // treat as stub only when all parts are stubs and the unit is still tiny,
  // or if we detect explicit stub markers inside function bodies.
  if (file.parseResult?.hasStubBodies) {
    file.isStub = true;
  } else if (file.lineCount > 50) {
    file.isStub = false;
  } else {
    file.isStub = allPartStub || anyPartStub;
  }
}

/**
 * Parse all files in a directory (or a single file) into logical units.
 */
export function parseDirectory(
  dirPath: string,
  language: Language,
  _extensions?: string[],
): Map<string, FileEntry> {
  const files = new Map<string, FileEntry>();

  const stat = fs.statSync(dirPath);
  if (stat.isFile()) {
    if (!shouldParseFile(dirPath, language)) {
      return files;
    }

    const filename = path.basename(dirPath);
    const relPath = filename;
    const ext = path.extname(filename);
    const stem = path.basename(filename, ext);

    const entry: FileEntry = {
      filename,
      filepath: dirPath,
      relativePath: relPath,
      language,
      paths: [dirPath],
      stem,
      qualifiedName: makeQualifiedName(relPath),
      parseResult: null,
      imports: [],
      exports: [],
      lineCount: 0,
      isStub: false,
    };

    parseLogicalEntry(entry);
    files.set(stem, entry);
    return files;
  }

  for (const fullPath of collectFiles(dirPath)) {
    if (!shouldParseFile(fullPath, language)) continue;

    const relPath = normalizeRelPath(path.relative(dirPath, fullPath));
    const ext = path.extname(fullPath);
    const stem = path.basename(fullPath, ext);
    const stemNormalized = normalizeStem(stem);

    const relDir = normalizeRelPath(path.dirname(relPath));
    const logicalKey = relDir === "." ? stemNormalized : `${relDir}/${stemNormalized}`;

    const existing = files.get(logicalKey);
    if (existing) {
      existing.paths.push(fullPath);

      if (HEADER_EXTENSIONS.has(ext)) {
        existing.filename = path.basename(fullPath);
        existing.filepath = fullPath;
        existing.relativePath = relPath;
      }
      continue;
    }

    files.set(logicalKey, {
      filename: path.basename(fullPath),
      filepath: fullPath,
      relativePath: relPath,
      language,
      paths: [fullPath],
      stem: stemNormalized,
      qualifiedName: makeQualifiedName(relPath),
      parseResult: null,
      imports: [],
      exports: [],
      lineCount: 0,
      isStub: false,
    });
  }

  for (const file of files.values()) {
    parseLogicalEntry(file);
  }

  return files;
}

/**
 * Build dependency graph from file entries.
 */
export function buildDepGraph(files: Map<string, FileEntry>): DepGraph {
  const nodes = new Map<string, DepNode>();

  for (const [relPath] of files.entries()) {
    nodes.set(relPath, {
      relativePath: relPath,
      dependents: [],
      dependencies: [],
    });
  }

  for (const [relPath, file] of files.entries()) {
    for (const imp of file.imports) {
      const impNorm = normalizeName(imp.replace(/\.(ts|tsx|rs|kt|kts|cpp|cc|cxx|hpp|h)$/g, ""));

      for (const [otherPath, otherFile] of files.entries()) {
        if (relPath === otherPath) continue;

        const otherStem = normalizeName(otherFile.stem);
        const otherPathNorm = normalizeName(otherPath.replace(/\.(ts|tsx|rs|kt|kts|cpp|cc|cxx|hpp|h)$/g, ""));

        if (impNorm.includes(otherStem) || impNorm.includes(otherPathNorm)) {
          const node = nodes.get(relPath);
          const otherNode = nodes.get(otherPath);
          if (!node || !otherNode) continue;

          if (!node.dependencies.includes(otherPath)) {
            node.dependencies.push(otherPath);
          }
          if (!otherNode.dependents.includes(relPath)) {
            otherNode.dependents.push(relPath);
          }
        }
      }
    }
  }

  return { nodes };
}

/**
 * Rank files by porting priority based on dependents and similarity.
 */
export function rankPortingPriority(
  srcFiles: Map<string, FileEntry>,
  tgtFiles: Map<string, FileEntry>,
  depGraph: DepGraph,
): Array<{ path: string; priority: number; dependents: number }> {
  const rankings: Array<{ path: string; priority: number; dependents: number }> = [];

  const targetByStem = new Set<string>();
  for (const target of tgtFiles.values()) {
    targetByStem.add(normalizeName(target.stem));
  }

  for (const [srcPath, srcFile] of srcFiles.entries()) {
    let isPorted = false;

    if (tgtFiles.has(srcPath)) {
      isPorted = true;
    } else if (targetByStem.has(normalizeName(srcFile.stem))) {
      isPorted = true;
    } else {
      for (const tgtFile of tgtFiles.values()) {
        if (tgtFile.parseResult?.portLintHeader?.sourcePath === srcPath) {
          isPorted = true;
          break;
        }
      }
    }

    if (isPorted) continue;

    const node = depGraph.nodes.get(srcPath);
    const dependents = node?.dependents.length || 0;

    rankings.push({
      path: srcPath,
      priority: dependents,
      dependents,
    });
  }

  rankings.sort((a, b) => b.priority - a.priority);
  return rankings;
}

/**
 * Find missing files in target directory.
 */
export function findMissingFiles(
  srcFiles: Map<string, FileEntry>,
  tgtFiles: Map<string, FileEntry>,
): Array<{
  path: string;
  matchedBy: "header" | "name" | "none";
  priority: number;
}> {
  const missing: Array<{
    path: string;
    matchedBy: "header" | "name" | "none";
    priority: number;
  }> = [];

  const portedByHeader = new Set<string>();
  const targetByStem = new Set<string>();

  for (const tgtFile of tgtFiles.values()) {
    targetByStem.add(normalizeName(tgtFile.stem));

    if (tgtFile.parseResult?.portLintHeader) {
      portedByHeader.add(tgtFile.parseResult.portLintHeader.sourcePath);
    }
  }

  for (const [srcPath, srcFile] of srcFiles.entries()) {
    if (tgtFiles.has(srcPath)) continue;
    if (portedByHeader.has(srcPath)) continue;
    if (targetByStem.has(normalizeName(srcFile.stem))) continue;

    missing.push({
      path: srcPath,
      matchedBy: "none",
      priority: 0,
    });
  }

  return missing;
}

/**
 * Print codebase statistics.
 */
export function printCodebaseStats(files: Map<string, FileEntry>): void {
  console.log("=== Codebase Statistics ===\n");

  let totalFiles = 0;
  let totalLines = 0;
  let stubFiles = 0;
  let todoCount = 0;

  for (const file of files.values()) {
    totalFiles++;
    totalLines += file.lineCount;
    if (file.isStub) stubFiles++;

    for (const filePath of file.paths) {
      const content = fs.readFileSync(filePath, "utf-8");
      const todoMatches = content.match(/TODO/gi);
      if (todoMatches) {
        todoCount += todoMatches.length;
      }
    }
  }

  console.log(`Total files:       ${totalFiles}`);
  console.log(`Total lines:       ${totalLines}`);
  console.log(`Stub files:        ${stubFiles}`);
  console.log(`TODO comments:     ${todoCount}`);
  console.log(`Avg lines/file:    ${(totalLines / Math.max(totalFiles, 1)).toFixed(1)}\n`);
}
