export {
  Language,
  NodeType,
  TreeNode,
  CommentStats,
  ParseResult,
  SimilarityMetrics,
  DocumentationComparison,
  PortLintHeader,
} from "./types.js";
export { parseFile, parseString } from "./ast-parser.js";
export {
  computeSimilarity,
  histogramCosineSimilarity,
  histogramCosineSimilarityMacro,
  combinedSimilarityWithContent,
  identifierCosineSimilarity,
  canonicalIdentifierCosineSimilarity,
  canonicalIdentifierJaccardSimilarity,
  docCosineSimilarity,
  compareDocumentation,
  printSimilarityReport,
  formatSimilarity,
} from "./similarity.js";
export {
  parseDirectory,
  buildDepGraph,
  rankPortingPriority,
  findMissingFiles,
  printCodebaseStats,
} from "./codebase.js";
export {
  getNodeMap,
  TYPESCRIPT_NODE_MAP,
  RUST_NODE_MAP,
  KOTLIN_NODE_MAP,
  CPP_NODE_MAP,
} from "./node-maps.js";
export type { FileEntry, DepNode, DepGraph } from "./codebase.js";
export type { NodeMap as NodeTypeMap } from "./node-maps.js";
