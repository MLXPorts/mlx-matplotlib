import {
  TreeNode,
  NodeType,
  ParseResult,
  SimilarityMetrics,
  DocumentationComparison,
  IdentifierStats,
} from "./types.js";

const STRUCTURAL = new Set<NodeType>([
  NodeType.CLASS_DECLARATION,
  NodeType.CLASS_DEFINITION,
  NodeType.FUNCTION_DECLARATION,
  NodeType.FUNCTION_DEFINITION,
  NodeType.INTERFACE_DECLARATION,
  NodeType.ENUM_DECLARATION,
  NodeType.IF_STATEMENT,
  NodeType.FOR,
  NodeType.WHILE,
  NodeType.SWITCH,
  NodeType.TRY,
]);

const OPERATIONS = new Set<NodeType>([
  NodeType.CALL_EXPRESSION,
  NodeType.METHOD_EXPRESSION,
  NodeType.RETURN_STATEMENT,
  NodeType.THROW,
  NodeType.COMPARISON_OP,
  NodeType.LOGICAL_OP,
]);

const OPERATORS = new Set<NodeType>([
  NodeType.ARITHMETIC_OP,
  NodeType.BITWISE_OP,
  NodeType.ASSIGNMENT_OP,
]);

const COMMON = new Set<NodeType>([
  NodeType.IDENTIFIER,
  NodeType.VARIABLE_DECLARATION,
]);

function nodeWeight(nt: NodeType): number {
  if (STRUCTURAL.has(nt)) return 5.0;
  if (OPERATIONS.has(nt)) return 2.0;
  if (OPERATORS.has(nt)) return 1.5;
  if (COMMON.has(nt)) return 0.5;
  return 1.0;
}

/**
 * Compute cosine similarity between node histograms.
 */
function cosineSimilarity(
  vec1: Map<NodeType, number>,
  vec2: Map<NodeType, number>,
  macroFriendly: boolean = false,
): number {
  let dot = 0;
  let norm1 = 0;
  let norm2 = 0;

  const allKeys = new Set([...vec1.keys(), ...vec2.keys()]);

  for (const key of allKeys) {
    const v1 = vec1.get(key) || 0;
    const v2 = vec2.get(key) || 0;

    let weight = nodeWeight(key);
    if (macroFriendly) {
      weight = key === NodeType.IDENTIFIER || key === NodeType.UNKNOWN ? 1.0 : 0.0;
    }

    const w1 = weight * v1;
    const w2 = weight * v2;

    dot += w1 * w2;
    norm1 += w1 * w1;
    norm2 += w2 * w2;
  }

  if (norm1 < 1e-8 || norm2 < 1e-8) return 0;
  return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

function cosineSimilarityStringMap(
  vec1: Map<string, number>,
  vec2: Map<string, number>,
): number {
  let dot = 0;
  let norm1 = 0;
  let norm2 = 0;

  const allKeys = new Set([...vec1.keys(), ...vec2.keys()]);
  for (const key of allKeys) {
    const v1 = vec1.get(key) || 0;
    const v2 = vec2.get(key) || 0;

    dot += v1 * v2;
    norm1 += v1 * v1;
    norm2 += v2 * v2;
  }

  if (norm1 < 1e-8 || norm2 < 1e-8) return 0;
  return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * Compute Jaccard similarity between two sets.
 */
function jaccardSimilarity(set1: Set<NodeType>, set2: Set<NodeType>): number {
  const intersection = new Set<NodeType>();
  const union = new Set<NodeType>();

  for (const item of set1) {
    if (set2.has(item)) {
      intersection.add(item);
    }
    union.add(item);
  }

  for (const item of set2) {
    union.add(item);
  }

  if (union.size === 0) return 1;
  return intersection.size / union.size;
}

/**
 * Extract node types as a set.
 */
function extractNodeTypes(tree: TreeNode): Set<NodeType> {
  const types = new Set<NodeType>();

  function walk(node: TreeNode) {
    types.add(node.nodeType);
    for (const child of node.children) {
      walk(child);
    }
  }

  walk(tree);
  return types;
}

/**
 * Compare tree shape (size and depth).
 */
function structureSimilarity(tree1: TreeNode, tree2: TreeNode): number {
  const size1 = tree1.size();
  const size2 = tree2.size();
  const depth1 = tree1.depth();
  const depth2 = tree2.depth();

  const maxSize = Math.max(size1, size2);
  const maxDepth = Math.max(depth1, depth2);

  const sizeSim = maxSize > 0 ? 1 - Math.abs(size1 - size2) / maxSize : 1;
  const depthSim = maxDepth > 0 ? 1 - Math.abs(depth1 - depth2) / maxDepth : 1;

  return (sizeSim + depthSim) / 2;
}

function cloneTree(node: TreeNode): TreeNode {
  const clone = new TreeNode(node.nodeType, node.label);
  clone.startPosition = { ...node.startPosition };
  clone.endPosition = { ...node.endPosition };
  clone.children = node.children.map((child) => cloneTree(child));
  return clone;
}

function combinedShapeSimilarity(
  result1: ParseResult,
  result2: ParseResult,
  macroFriendly: boolean = false,
): SimilarityMetrics {
  const tree1 = cloneTree(result1.tree);
  const tree2 = cloneTree(result2.tree);

  tree1.flattenNodeType(NodeType.PACKAGE);
  tree2.flattenNodeType(NodeType.PACKAGE);

  const cosineHistogram = cosineSimilarity(
    result1.nodeTypes,
    result2.nodeTypes,
    macroFriendly,
  );
  const structure = structureSimilarity(tree1, tree2);

  const nodeTypes1 = extractNodeTypes(tree1);
  const nodeTypes2 = extractNodeTypes(tree2);
  const jaccard = jaccardSimilarity(nodeTypes1, nodeTypes2);

  const combined = cosineHistogram * 0.5 + structure * 0.3 + jaccard * 0.2;

  return {
    cosineHistogram,
    structure,
    jaccard,
    combined,
  };
}

export function histogramCosineSimilarity(
  result1: ParseResult,
  result2: ParseResult,
): number {
  return cosineSimilarity(result1.nodeTypes, result2.nodeTypes, false);
}

export function histogramCosineSimilarityMacro(
  result1: ParseResult,
  result2: ParseResult,
): number {
  return cosineSimilarity(result1.nodeTypes, result2.nodeTypes, true);
}

export function identifierCosineSimilarity(
  stats1: IdentifierStats,
  stats2: IdentifierStats,
): number {
  return cosineSimilarityStringMap(stats1.identifierFreq, stats2.identifierFreq);
}

export function canonicalIdentifierCosineSimilarity(
  stats1: IdentifierStats,
  stats2: IdentifierStats,
): number {
  return cosineSimilarityStringMap(stats1.canonicalFreq, stats2.canonicalFreq);
}

export function canonicalIdentifierJaccardSimilarity(
  stats1: IdentifierStats,
  stats2: IdentifierStats,
): number {
  const set1 = new Set(stats1.canonicalFreq.keys());
  const set2 = new Set(stats2.canonicalFreq.keys());

  if (set1.size === 0 && set2.size === 0) return 1;
  if (set1.size === 0 || set2.size === 0) return 0;

  let intersection = 0;
  for (const id of set1) {
    if (set2.has(id)) intersection++;
  }

  const union = set1.size + set2.size - intersection;
  if (union === 0) return 1;
  return intersection / union;
}

/**
 * Compute shape-only similarity (no identifier content).
 */
export function computeSimilarity(
  result1: ParseResult,
  result2: ParseResult,
  options?: { macroFriendly?: boolean },
): SimilarityMetrics {
  return combinedShapeSimilarity(result1, result2, options?.macroFriendly ?? false);
}

/**
 * Compute content-aware similarity with identifier dominance.
 */
export function combinedSimilarityWithContent(
  result1: ParseResult,
  result2: ParseResult,
  options?: { macroFriendly?: boolean },
): number {
  const shape = combinedShapeSimilarity(result1, result2, options?.macroFriendly ?? false);

  const idCos = canonicalIdentifierCosineSimilarity(
    result1.identifierStats,
    result2.identifierStats,
  );
  const idJaccard = canonicalIdentifierJaccardSimilarity(
    result1.identifierStats,
    result2.identifierStats,
  );

  return (
    0.5 * idCos +
    0.15 * idJaccard +
    0.15 * shape.cosineHistogram +
    0.1 * shape.jaccard +
    0.1 * shape.structure
  );
}

/**
 * Compute cosine similarity of doc word frequencies.
 */
export function docCosineSimilarity(
  freq1: Map<string, number>,
  freq2: Map<string, number>,
): number {
  if (freq1.size === 0 || freq2.size === 0) return 0;
  return cosineSimilarityStringMap(freq1, freq2);
}

/**
 * Compare documentation between two parse results.
 */
export function compareDocumentation(
  result1: ParseResult,
  result2: ParseResult,
): DocumentationComparison {
  const docCommentDifference =
    result2.commentStats.docCommentCount - result1.commentStats.docCommentCount;

  const docLineDifference =
    result2.commentStats.totalDocLines - result1.commentStats.totalDocLines;

  const docTextCosine = docCosineSimilarity(
    result1.commentStats.wordFreq,
    result2.commentStats.wordFreq,
  );

  return {
    docCommentDifference,
    docLineDifference,
    docTextCosine,
  };
}

/**
 * Format similarity as percentage string.
 */
export function formatSimilarity(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

/**
 * Print similarity report to console.
 */
export function printSimilarityReport(
  result1: ParseResult,
  result2: ParseResult,
  options?: { macroFriendly?: boolean; includeContent?: boolean },
): void {
  console.log("=== AST Similarity Report ===");
  console.log(`File 1: ${result1.filename} (${result1.language})`);
  console.log(
    `  Tree size: ${result1.tree.size()}, depth: ${result1.tree.depth()}\n`,
  );

  console.log(`File 2: ${result2.filename} (${result2.language})`);
  console.log(
    `  Tree size: ${result2.tree.size()}, depth: ${result2.tree.depth()}\n`,
  );

  const metrics = computeSimilarity(result1, result2, {
    macroFriendly: options?.macroFriendly,
  });

  console.log("Similarity Metrics:");
  console.log(
    `  Cosine (histogram):    ${formatSimilarity(metrics.cosineHistogram)}`,
  );
  console.log(`  Structure:             ${formatSimilarity(metrics.structure)}`);
  console.log(`  Jaccard:               ${formatSimilarity(metrics.jaccard)}`);
  console.log(`  Combined Score:        ${formatSimilarity(metrics.combined)}\n`);

  if (options?.includeContent) {
    const contentScore = combinedSimilarityWithContent(result1, result2, {
      macroFriendly: options?.macroFriendly,
    });
    console.log(`Content-Aware Score:     ${formatSimilarity(contentScore)}\n`);
  }

  const docComparison = compareDocumentation(result1, result2);
  console.log("=== Documentation Comparison ===");
  console.log(
    `  Doc count:    ${result1.commentStats.docCommentCount} vs ${result2.commentStats.docCommentCount} (diff: ${docComparison.docCommentDifference})`,
  );
  console.log(
    `  Doc lines:    ${result1.commentStats.totalDocLines} vs ${result2.commentStats.totalDocLines} (diff: ${docComparison.docLineDifference})`,
  );
  console.log(
    `  Doc cosine:   ${formatSimilarity(docComparison.docTextCosine)}\n`,
  );
}
