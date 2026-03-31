/**
 * Language support for AST parsing
 */
export enum Language {
  TYPESCRIPT = "typescript",
  RUST = "rust",
  KOTLIN = "kotlin",
  CPP = "cpp",
  C = "c",
}

/**
 * Normalized AST node types across languages
 */
export enum NodeType {
  // Root nodes
  SOURCE_FILE = "source_file",
  MODULE = "module",

  // Declarations
  FUNCTION_DECLARATION = "function_declaration",
  CLASS_DECLARATION = "class_declaration",
  INTERFACE_DECLARATION = "interface_declaration",
  TYPE_DECLARATION = "type_declaration",
  ENUM_DECLARATION = "enum_declaration",
  VARIABLE_DECLARATION = "variable_declaration",

  // Definitions
  FUNCTION_DEFINITION = "function_definition",
  CLASS_DEFINITION = "class_definition",
  METHOD_DEFINITION = "method_definition",

  // Statements
  BLOCK = "block",
  IF_STATEMENT = "if_statement",
  FOR_STATEMENT = "for_statement",
  WHILE_STATEMENT = "while_statement",
  RETURN_STATEMENT = "return_statement",
  EXPRESSION_STATEMENT = "expression_statement",
  ASSIGNMENT = "assignment",

  // Expressions
  CALL_EXPRESSION = "call_expression",
  MEMBER_EXPRESSION = "member_expression",
  BINARY_EXPRESSION = "binary_expression",
  UNARY_EXPRESSION = "unary_expression",
  LITERAL = "literal",
  IDENTIFIER = "identifier",

  // Parameters
  PARAMETER = "parameter",
  PARAMETER_LIST = "parameter_list",

  // Type annotations
  TYPE_ANNOTATION = "type_annotation",
  TYPE_IDENTIFIER = "type_identifier",

  // Other
  IMPORT_STATEMENT = "import_statement",
  EXPORT_STATEMENT = "export_statement",
  COMMENT = "comment",
  PROPERTY = "property",
  PACKAGE = "package",
  ANNOTATION = "annotation",
  MODIFIER = "modifier",
  OTHER = "other",

  // Additional node types for cross-language support
  SWITCH = "switch",
  BREAK = "break",
  CONTINUE = "continue",
  THROW = "throw",
  TRY = "try",
  FOR = "for",
  WHILE = "while",
  METHOD_EXPRESSION = "method_expression",
  INDEX = "index",
  ARITHMETIC_OP = "arithmetic_op",
  COMPARISON_OP = "comparison_op",
  LOGICAL_OP = "logical_op",
  BITWISE_OP = "bitwise_op",
  ASSIGNMENT_OP = "assignment_op",

  // Unknown/Fallback
  UNKNOWN = "unknown",
}

/**
 * Tree node representing an AST element
 */
export class TreeNode {
  nodeType: NodeType;
  label: string;
  children: TreeNode[];
  startPosition: { row: number; column: number };
  endPosition: { row: number; column: number };

  constructor(nodeType: NodeType, label: string, children: TreeNode[] = []) {
    this.nodeType = nodeType;
    this.label = label;
    this.children = children;
    this.startPosition = { row: 0, column: 0 };
    this.endPosition = { row: 0, column: 0 };
  }

  isLeaf(): boolean {
    return this.children.length === 0;
  }

  size(): number {
    return 1 + this.children.reduce((sum, child) => sum + child.size(), 0);
  }

  depth(): number {
    if (this.isLeaf()) return 0;
    return 1 + Math.max(...this.children.map((child) => child.depth()));
  }

  /**
   * Flatten nodes of specific type (replacing them with their children)
   * Used to reduce structural noise in AST comparisons
   */
  flattenNodeType(typeToFlatten: NodeType): void {
    if (this.children.length === 0) return;

    const newChildren: TreeNode[] = [];

    for (const child of this.children) {
      // Recurse first (bottom-up flattening)
      child.flattenNodeType(typeToFlatten);

      if (child.nodeType === typeToFlatten) {
        // Dissolve this node, append its children to current parent
        newChildren.push(...child.children);
      } else {
        newChildren.push(child);
      }
    }

    this.children = newChildren;
  }
}

/**
 * Statistics about comments/documentation
 */
export interface CommentStats {
  docCommentCount: number;
  lineCommentCount: number;
  blockCommentCount: number;
  totalCommentLines: number;
  totalDocLines: number;
  docTexts: string[];
  wordFreq: Map<string, number>;
}

/**
 * Identifier statistics used for content-aware similarity.
 */
export interface IdentifierStats {
  identifierFreq: Map<string, number>;
  canonicalFreq: Map<string, number>;
  totalIdentifiers: number;
}

/**
 * AST parse result with metadata
 */
export interface ParseResult {
  tree: TreeNode;
  filename: string;
  language: Language;
  commentStats: CommentStats;
  identifierStats: IdentifierStats;
  hasStubBodies: boolean;
  nodeTypes: Map<NodeType, number>; // Histogram
  importPaths: string[];
  exportPaths: string[];
  portLintHeader?: PortLintHeader;
}

/**
 * Similarity metrics between two ASTs
 */
export interface SimilarityMetrics {
  cosineHistogram: number;
  structure: number;
  jaccard: number;
  combined: number;
}

/**
 * Documentation comparison
 */
export interface DocumentationComparison {
  docCommentDifference: number;
  docLineDifference: number;
  docTextCosine: number;
}

/**
 * Port-lint header information
 */
export interface PortLintHeader {
  sourcePath: string;
  matchedBy: "header" | "name";
  confidence: number;
}
