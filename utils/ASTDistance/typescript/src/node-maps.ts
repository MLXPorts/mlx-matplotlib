import { Language, NodeType } from "./types.js";

/**
 * Normalized node type mappings from C++ implementation
 * These provide comprehensive coverage for all supported languages
 */

export type NodeMap = { [key: string]: NodeType };

/**
 * TypeScript node type mappings
 */
export const TYPESCRIPT_NODE_MAP: NodeMap = {
  // Statements
  source_file: NodeType.SOURCE_FILE,
  program: NodeType.SOURCE_FILE,
  block: NodeType.BLOCK,
  statement_block: NodeType.BLOCK,
  if_statement: NodeType.IF_STATEMENT,
  for_statement: NodeType.FOR,
  for_in_statement: NodeType.FOR,
  for_of_statement: NodeType.FOR,
  while_statement: NodeType.WHILE,
  do_statement: NodeType.WHILE,
  switch_statement: NodeType.SWITCH,
  switch_case: NodeType.SWITCH,
  case: NodeType.SWITCH,
  break_statement: NodeType.BREAK,
  continue_statement: NodeType.CONTINUE,
  return_statement: NodeType.RETURN_STATEMENT,
  throw_statement: NodeType.THROW,
  try_statement: NodeType.TRY,
  catch_clause: NodeType.TRY,

  // Assignments
  assignment_expression: NodeType.ASSIGNMENT,
  augmented_assignment_expression: NodeType.ASSIGNMENT,

  // Comparisons (need operator checking)
  binary_expression: NodeType.UNKNOWN,

  // Arithmetic
  unary_expression: NodeType.UNARY_EXPRESSION,
  update_expression: NodeType.UNARY_EXPRESSION,

  // Literals
  identifier: NodeType.IDENTIFIER,
  property_identifier: NodeType.IDENTIFIER,
  type_identifier: NodeType.IDENTIFIER,
  number: NodeType.LITERAL,
  string: NodeType.LITERAL,
  template_string: NodeType.LITERAL,
  regex: NodeType.LITERAL,
  true: NodeType.LITERAL,
  false: NodeType.LITERAL,
  null: NodeType.LITERAL,
  undefined: NodeType.LITERAL,

  // Expressions
  call_expression: NodeType.CALL_EXPRESSION,
  new_expression: NodeType.CALL_EXPRESSION,
  member_expression: NodeType.MEMBER_EXPRESSION,
  subscript_expression: NodeType.INDEX,
  array: NodeType.UNKNOWN,
  object: NodeType.UNKNOWN,

  // Declarations
  function_declaration: NodeType.FUNCTION_DECLARATION,
  function_expression: NodeType.FUNCTION_DECLARATION,
  arrow_function: NodeType.FUNCTION_DECLARATION,
  method_definition: NodeType.METHOD_DEFINITION,
  class_declaration: NodeType.CLASS_DECLARATION,
  class_expression: NodeType.CLASS_DECLARATION,
  class_body: NodeType.BLOCK,
  interface_declaration: NodeType.INTERFACE_DECLARATION,
  type_alias_declaration: NodeType.TYPE_DECLARATION,
  enum_declaration: NodeType.ENUM_DECLARATION,
  lexical_declaration: NodeType.VARIABLE_DECLARATION,
  variable_declaration: NodeType.VARIABLE_DECLARATION,
  formal_parameters: NodeType.PARAMETER_LIST,
  required_parameter: NodeType.PARAMETER,
  optional_parameter: NodeType.PARAMETER,
  rest_parameter: NodeType.PARAMETER,
  type_parameter: NodeType.PARAMETER,

  // Types
  type_annotation: NodeType.TYPE_ANNOTATION,
  predefined_type: NodeType.TYPE_IDENTIFIER,
  union_type: NodeType.TYPE_DECLARATION,
  intersection_type: NodeType.TYPE_DECLARATION,
  array_type: NodeType.TYPE_DECLARATION,
  array_pattern: NodeType.TYPE_DECLARATION,
  generic_type: NodeType.TYPE_DECLARATION,
  type_arguments: NodeType.TYPE_DECLARATION,

  // Import/Export
  import_statement: NodeType.IMPORT_STATEMENT,
  export_statement: NodeType.EXPORT_STATEMENT,

  // Decorators
  decorator: NodeType.UNKNOWN,

  // Comments
  comment: NodeType.COMMENT,
  jsdoc: NodeType.COMMENT,
  line_comment: NodeType.COMMENT,
  block_comment: NodeType.COMMENT,

  // Properties
  property_signature: NodeType.PROPERTY,
  property_definition: NodeType.PROPERTY,

  // Keywords
  this: NodeType.IDENTIFIER,
  super: NodeType.IDENTIFIER,
};

/**
 * Rust node type mappings (comprehensive)
 */
export const RUST_NODE_MAP: NodeMap = {
  // Statements
  source_file: NodeType.SOURCE_FILE,
  mod_item: NodeType.MODULE,
  block: NodeType.BLOCK,
  block_expression: NodeType.BLOCK,
  if_expression: NodeType.IF_STATEMENT,
  if_let_expression: NodeType.IF_STATEMENT,
  for_expression: NodeType.FOR,
  while_expression: NodeType.WHILE,
  while_let_expression: NodeType.WHILE,
  loop_expression: NodeType.WHILE,
  match_expression: NodeType.SWITCH,
  match_arm: NodeType.SWITCH,
  match_block: NodeType.BLOCK,
  return_expression: NodeType.RETURN_STATEMENT,
  continue_expression: NodeType.CONTINUE,
  break_expression: NodeType.BREAK,
  try_expression: NodeType.TRY,
  try_block: NodeType.TRY,

  // Assignments
  assignment_expression: NodeType.ASSIGNMENT,
  compound_assignment_expr: NodeType.ASSIGNMENT,

  // Comparisons
  binary_expression: NodeType.UNKNOWN,

  // Literals
  identifier: NodeType.IDENTIFIER,
  integer_literal: NodeType.LITERAL,
  float_literal: NodeType.LITERAL,
  string_literal: NodeType.LITERAL,
  raw_string_literal: NodeType.LITERAL,
  boolean_literal: NodeType.LITERAL,
  char_literal: NodeType.LITERAL,

  // Expressions
  call_expression: NodeType.CALL_EXPRESSION,
  method_call_expression: NodeType.METHOD_EXPRESSION,
  field_expression: NodeType.MEMBER_EXPRESSION,
  index_expression: NodeType.INDEX,
  array_expression: NodeType.UNKNOWN,
  tuple_expression: NodeType.UNKNOWN,
  struct_expression: NodeType.UNKNOWN,
  parenthesized_expression: NodeType.UNKNOWN,

  // Declarations
  function_item: NodeType.FUNCTION_DECLARATION,
  struct_item: NodeType.CLASS_DECLARATION,
  enum_item: NodeType.ENUM_DECLARATION,
  trait_item: NodeType.INTERFACE_DECLARATION,
  impl_item: NodeType.CLASS_DECLARATION,
  let_declaration: NodeType.VARIABLE_DECLARATION,
  static_item: NodeType.VARIABLE_DECLARATION,
  const_item: NodeType.VARIABLE_DECLARATION,
  type_alias_item: NodeType.TYPE_DECLARATION,
  parameter: NodeType.PARAMETER,
  type_parameter: NodeType.PARAMETER,

  // Types
  type_identifier: NodeType.TYPE_IDENTIFIER,
  array_type: NodeType.TYPE_DECLARATION,
  generic_type: NodeType.TYPE_DECLARATION,
  reference_type: NodeType.TYPE_DECLARATION,
  pointer_type: NodeType.TYPE_DECLARATION,
  slice_type: NodeType.TYPE_DECLARATION,
  tuple_type: NodeType.TYPE_DECLARATION,
  inferred_type: NodeType.TYPE_DECLARATION,

  // Import/Use
  use_declaration: NodeType.IMPORT_STATEMENT,
  mod_declaration: NodeType.IMPORT_STATEMENT,

  // Attributes
  attribute_item: NodeType.UNKNOWN,
  inner_attribute_item: NodeType.UNKNOWN,

  // Comments
  line_comment: NodeType.COMMENT,
  block_comment: NodeType.COMMENT,

  // Visibility
  visibility_modifier: NodeType.UNKNOWN,
};

/**
 * Kotlin node type mappings
 */
export const KOTLIN_NODE_MAP: NodeMap = {
  // Statements
  kotlin_file: NodeType.SOURCE_FILE,
  statements: NodeType.BLOCK,
  control_structure_body: NodeType.BLOCK,
  block: NodeType.BLOCK,
  if_expression: NodeType.IF_STATEMENT,
  for_statement: NodeType.FOR,
  while_statement: NodeType.WHILE,
  do_while_statement: NodeType.WHILE,
  when_expression: NodeType.SWITCH,
  when_entry: NodeType.SWITCH,
  jump_expression: NodeType.RETURN_STATEMENT,
  try_expression: NodeType.TRY,

  // Assignments
  assignment: NodeType.ASSIGNMENT,
  directly_assignable_expression: NodeType.ASSIGNMENT,

  // Literals
  simple_identifier: NodeType.IDENTIFIER,
  integer_literal: NodeType.LITERAL,
  long_literal: NodeType.LITERAL,
  real_literal: NodeType.LITERAL,
  float_literal: NodeType.LITERAL,
  string_literal: NodeType.LITERAL,
  character_literal: NodeType.LITERAL,
  boolean_literal: NodeType.LITERAL,
  null_literal: NodeType.LITERAL,
  line_string_literal: NodeType.LITERAL,
  multi_line_string_literal: NodeType.LITERAL,

  // Expressions
  call_expression: NodeType.CALL_EXPRESSION,
  navigation_expression: NodeType.MEMBER_EXPRESSION,
  indexing_expression: NodeType.INDEX,
  safe_navigation_expression: NodeType.MEMBER_EXPRESSION,

  // Declarations
  function_declaration: NodeType.FUNCTION_DECLARATION,
  class_declaration: NodeType.CLASS_DECLARATION,
  object_declaration: NodeType.CLASS_DECLARATION,
  companion_object: NodeType.CLASS_DECLARATION,
  enum_class: NodeType.ENUM_DECLARATION,
  enum_entry: NodeType.ENUM_DECLARATION,
  interface_declaration: NodeType.INTERFACE_DECLARATION,
  property_declaration: NodeType.VARIABLE_DECLARATION,
  val_declaration: NodeType.VARIABLE_DECLARATION,
  var_declaration: NodeType.VARIABLE_DECLARATION,
  parameter: NodeType.PARAMETER,
  type_parameter: NodeType.PARAMETER,

  // Types
  user_type: NodeType.TYPE_IDENTIFIER,
  nullable_type: NodeType.TYPE_DECLARATION,
  function_type: NodeType.TYPE_DECLARATION,
  type_projection: NodeType.TYPE_DECLARATION,

  // Import/Export
  import_header: NodeType.IMPORT_STATEMENT,
  import_list: NodeType.IMPORT_STATEMENT,
  package_header: NodeType.PACKAGE,

  // Annotations
  annotation: NodeType.ANNOTATION,
  annotation_entry: NodeType.ANNOTATION,

  // Modifiers
  modifier: NodeType.MODIFIER,
  modifiers: NodeType.MODIFIER,
  visibility_modifier: NodeType.MODIFIER,
  inheritance_modifier: NodeType.MODIFIER,
  function_modifier: NodeType.MODIFIER,
  platform_modifier: NodeType.MODIFIER,

  // Comments
  multiline_comment: NodeType.COMMENT,
  line_comment: NodeType.COMMENT,
  doc_comment: NodeType.COMMENT,

  // Additional constructs
  class_body: NodeType.BLOCK,
  function_body: NodeType.BLOCK,
  property_delegate: NodeType.OTHER,
  type_arguments: NodeType.TYPE_DECLARATION,
  value_arguments: NodeType.OTHER,
  this_expression: NodeType.IDENTIFIER,
  super_expression: NodeType.IDENTIFIER,

  // Lambdas
  lambda_literal: NodeType.FUNCTION_DECLARATION,
  anonymous_function: NodeType.FUNCTION_DECLARATION,
};

/**
 * C++ node type mappings
 */
export const CPP_NODE_MAP: NodeMap = {
  // Statements
  translation_unit: NodeType.SOURCE_FILE,
  compound_statement: NodeType.BLOCK,
  if_statement: NodeType.IF_STATEMENT,
  for_statement: NodeType.FOR,
  for_range_loop: NodeType.FOR,
  while_statement: NodeType.WHILE,
  do_statement: NodeType.WHILE,
  switch_statement: NodeType.SWITCH,
  case: NodeType.SWITCH,
  return_statement: NodeType.RETURN_STATEMENT,
  continue_statement: NodeType.CONTINUE,
  break_statement: NodeType.BREAK,
  try_statement: NodeType.TRY,
  throw_statement: NodeType.THROW,
  goto_statement: NodeType.BREAK,

  // Assignments
  assignment_expression: NodeType.ASSIGNMENT,
  compound_assignment_expression: NodeType.ASSIGNMENT,

  // Comparisons
  binary_expression: NodeType.UNKNOWN,
  conditional_expression: NodeType.UNKNOWN,

  // Literals
  identifier: NodeType.IDENTIFIER,
  field_identifier: NodeType.IDENTIFIER,
  namespace_identifier: NodeType.IDENTIFIER,
  type_identifier: NodeType.TYPE_IDENTIFIER,
  number_literal: NodeType.LITERAL,
  string_literal: NodeType.LITERAL,
  raw_string_literal: NodeType.LITERAL,
  char_literal: NodeType.LITERAL,
  true: NodeType.LITERAL,
  false: NodeType.LITERAL,
  nullptr: NodeType.LITERAL,
  null: NodeType.LITERAL,

  // Expressions
  call_expression: NodeType.CALL_EXPRESSION,
  field_expression: NodeType.MEMBER_EXPRESSION,
  subscript_expression: NodeType.INDEX,
  parenthesized_expression: NodeType.UNKNOWN,

  // Declarations
  function_definition: NodeType.FUNCTION_DECLARATION,
  function_declarator: NodeType.FUNCTION_DECLARATION,
  declaration: NodeType.VARIABLE_DECLARATION,
  class_specifier: NodeType.CLASS_DECLARATION,
  struct_specifier: NodeType.CLASS_DECLARATION,
  enum_specifier: NodeType.ENUM_DECLARATION,
  init_declarator: NodeType.VARIABLE_DECLARATION,
  parameter_declaration: NodeType.PARAMETER,
  template_parameter_list: NodeType.PARAMETER_LIST,

  // Types
  primitive_type: NodeType.TYPE_IDENTIFIER,
  qualified_identifier: NodeType.TYPE_IDENTIFIER,
  pointer_declarator: NodeType.TYPE_DECLARATION,
  reference_declarator: NodeType.TYPE_DECLARATION,
  array_declarator: NodeType.TYPE_DECLARATION,

  // Namespaces
  namespace_definition: NodeType.PACKAGE,
  declaration_list: NodeType.PACKAGE, // Map to PACKAGE to flatten structural nesting
  preproc_include: NodeType.IMPORT_STATEMENT,
  using_declaration: NodeType.IMPORT_STATEMENT,
  using_directive: NodeType.IMPORT_STATEMENT,

  // Templates
  template_declaration: NodeType.TYPE_DECLARATION,
  template_type: NodeType.TYPE_DECLARATION,
  template_argument_list: NodeType.TYPE_DECLARATION,

  // Lambdas
  lambda_expression: NodeType.FUNCTION_DECLARATION,

  // Modifiers and attributes
  storage_class_specifier: NodeType.MODIFIER,
  type_qualifier: NodeType.MODIFIER,
  virtual_specifier: NodeType.MODIFIER,
  access_specifier: NodeType.MODIFIER,
  attribute: NodeType.ANNOTATION,

  // Other constructs
  linkage_specification: NodeType.PACKAGE, // extern "C"
  base_clause: NodeType.TYPE_IDENTIFIER, // Inheritance list
  parameter_list: NodeType.OTHER,
  argument_list: NodeType.OTHER,
  field_declaration: NodeType.VARIABLE_DECLARATION,
  alias_declaration: NodeType.TYPE_IDENTIFIER,

  // Comments
  comment: NodeType.COMMENT,
  // Attributes not otherwise mapped
  attribute_declaration: NodeType.UNKNOWN,
};

/**
 * Get node type map for language
 */
export function getNodeMap(language: Language): NodeMap {
  switch (language) {
    case Language.TYPESCRIPT:
      return TYPESCRIPT_NODE_MAP;
    case Language.RUST:
      return RUST_NODE_MAP;
    case Language.KOTLIN:
      return KOTLIN_NODE_MAP;
    case Language.CPP:
    case Language.C:
      return CPP_NODE_MAP;
    default:
      return {};
  }
}
