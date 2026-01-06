import re
import uuid
from typing import Tuple, Union

from src.data_preprocessors.transformations.transformation_base import TransformationBase


class InlineSnippetInserter(TransformationBase):
    """C++-only snippet injector that inlines a benign helper lambda and spreads its use.

    Adds three lines inside the first detected function body:
    - a unique, no-op lambda definition mirroring the seed `add` helper
    - a benign call using the lambda (inline refactor: uses the helper directly without temp wrappers)
    - a void-cast of the result to preserve side-effect-free semantics

    The lines are spread near the start/middle/end of the function body when possible.
    """

    def __init__(
            self,
            parser_path: str,
            language: str,
            inline_func: str = (
                "int add(int a, int b)\n"
                "{\n"
                "    int sum = 0;\n"
                "    sum += a;\n"
                "    sum += b;\n"
                "    return sum;\n"
                "}"
            ),
        ):
        super().__init__(parser_path=parser_path, language=language)
        self.language = language
        self.inline_func = inline_func

    def _find_first_function(self, root) -> Union[None, object]:
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "function_definition":
                return node
            stack.extend(reversed(node.children))
        return None

    def _indent_for_body(self, code: str, brace_index: int) -> str:
        # Find the indentation level immediately after the opening brace
        newline_pos = code.rfind("\n", 0, brace_index + 1)
        if newline_pos == -1:
            return "    "
        indent = []
        i = newline_pos + 1
        while i < len(code) and code[i] in (" ", "\t"):
            indent.append(code[i])
            i += 1
        # default to 4 spaces if no indent detected
        return "".join(indent) if indent else "    "

    def _insert_at(self, code: str, inserts: Tuple[Tuple[int, str], ...]) -> str:
        # inserts: tuple of (position, text); apply from end to start to keep offsets valid
        result = code
        for pos, text in sorted(inserts, key=lambda x: x[0], reverse=True):
            result = result[:pos] + text + result[pos:]
        return result

    def _convert_helper_function(self, helper_code: str, indent: str, suffix: str) -> Union[str, None]:
        """Convert a helper function definition into an inline block using tree-sitter structure."""
        normalized_helper = helper_code.replace("{{", "{").replace("}}", "}")

        def slice_text(node):
            return normalized_helper[node.start_byte:node.end_byte]

        try:
            root = self.parse_code(normalized_helper)
            func = self._find_first_function(root)
            if func is None:
                return None

            decl_specs = None
            func_decl = None
            body = None
            for child in func.children:
                if child.type in {"declaration_specifiers", "primitive_type"}:
                    decl_specs = slice_text(child).strip()
                elif child.type.endswith("declarator"):
                    func_decl = child
                elif child.type == "compound_statement":
                    body = child
            if decl_specs is None or func_decl is None or body is None:
                return None

            params = []
            param_names = []
            param_list = None
            stack = [func_decl]
            while stack:
                node = stack.pop()
                if node.type == "parameter_list":
                    param_list = node
                    break
                stack.extend(reversed(node.children))

            if param_list:
                for p in param_list.children:
                    if p.type != "parameter_declaration":
                        continue
                    text = slice_text(p).strip().rstrip(',').strip()
                    if not text.endswith(";"):
                        text = text + ";"
                    params.append(f"{indent}{text}")
                    pname = None
                    for d in p.children:
                        if d.type == "identifier":
                            pname = slice_text(d)
                            break
                    if pname:
                        param_names.append(pname)

            result_var = "result"
            has_result = decl_specs.strip() != "void"
            first_param = param_names[0] if param_names else None
            def normalize_op_spacing(text: str) -> str:
                return text

            lines = []
            if has_result:
                lines.append(f"{indent}{decl_specs} {result_var};")
            lines.extend(params)
            lines.append("")

            def emit_stmt(node):
                ntype = node.type

                if ntype == "return_statement":
                    expr_child = None
                    for c in node.children:
                        if c.type not in {"return", ";"}:
                            expr_child = c
                            break
                    expr_text = slice_text(expr_child).strip() if expr_child else ""
                    if has_result and expr_text:
                        lines.append(f"{indent}{result_var} = {expr_text};")
                    return

                if ntype == "for_statement":
                    text = slice_text(node)
                    lpar = text.find("(")
                    rpar = text.find(")")
                    header = text[:rpar + 1] if lpar != -1 and rpar != -1 else text
                    lines.append(f"{indent}{header}")

                    body_child = None
                    for c in node.children:
                        if c.type == "compound_statement":
                            body_child = c
                            break
                    if body_child and body_child.type == "compound_statement":
                        lines.append(f"{indent}{{")
                        for st in body_child.children:
                            if st.type in {"{", "}"}:
                                continue
                            emit_stmt(st)
                        lines.append(f"{indent}}}")
                    elif body_child:
                        emit_stmt(body_child)
                    return

                if ntype == "declaration":
                    decl_text = slice_text(node).strip()
                    if not decl_text.endswith(";"):
                        decl_text += ";"
                    lines.append(f"{indent}{decl_text}")
                    return

                if ntype == "expression_statement":
                    expr_text = slice_text(node).strip()
                    if not expr_text.endswith(";"):
                        expr_text += ";"
                    lines.append(f"{indent}{expr_text}")
                    return

                if ntype == "compound_statement":
                    for st in node.children:
                        if st.type in {"{", "}"}:
                            continue
                        emit_stmt(st)
                    return

                text = slice_text(node).strip()
                    
                if text:
                    if not text.endswith(";"):
                        text += ";"
                    lines.append(f"{indent}{text}")

            for stmt in body.children:
                if stmt.type in {"{", "}"}:
                    continue
                emit_stmt(stmt)

            return "\n".join(lines)
        except Exception:
            return None

    def transform_code(self, code: Union[str, bytes]) -> Tuple[str, object]:
        if self.language != "cpp":
            return code if isinstance(code, str) else code.decode(), {"success": False, "reason": "unsupported_language"}

        source = code if isinstance(code, str) else code.decode()
        try:
            root = self.parse_code(source)
            func = self._find_first_function(root)
            if func is None:
                return source, {"success": False, "reason": "no_function"}

            body = None
            for child in func.children:
                if child.type == "compound_statement":
                    body = child
                    break
            if body is None:
                return source, {"success": False, "reason": "no_body"}

            open_brace_idx = source.find("{", body.start_byte, body.start_byte + 5)
            close_brace_idx = source.rfind("}", body.start_byte, body.end_byte)
            if open_brace_idx == -1 or close_brace_idx == -1:
                return source, {"success": False, "reason": "brace_not_found"}

            indent = self._indent_for_body(source, open_brace_idx)
            suffix = uuid.uuid4().hex[:8]
            helper_code = self.inline_func
            if not helper_code:
                return source, {"success": False, "reason": "no_helper_template"}

            converted_helper = self._convert_helper_function(helper_code, indent, suffix)
            if converted_helper is None:
                return source, {"success": False, "reason": "convert_failed"}

            helper_decl = converted_helper
            call_stmt = ""

            inserts = []
            if helper_decl.strip():
                lambda_line = f"\n{helper_decl}\n"
                inserts.append((open_brace_idx + 1, lambda_line))

            # If no inserts added, return original with failure metadata
            if not inserts:
                return source, {"success": False, "reason": "no_inserts"}

            transformed = self._insert_at(source, tuple(inserts))
            return transformed, {"success": True, "reason": "ok"}
        except Exception as exc:  # pragma: no cover - defensive
            return source, {"success": False, "reason": str(exc)}
