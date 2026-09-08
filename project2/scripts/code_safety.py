from __future__ import annotations

import ast
from dataclasses import asdict, dataclass


ALLOWED_IMPORT_ROOTS = {"cadquery", "math"}
DENIED_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
DENIED_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
)


@dataclass
class SafetyIssue:
    line: int
    column: int
    code: str
    message: str


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[SafetyIssue] = []

    def add(self, node: ast.AST, code: str, message: str) -> None:
        self.issues.append(
            SafetyIssue(
                line=getattr(node, "lineno", 0),
                column=getattr(node, "col_offset", 0),
                code=code,
                message=message,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self.add(node, "unsafe_import", f"禁止导入 {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            self.add(node, "unsafe_import", f"禁止从 {node.module} 导入")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
            self.add(node, "unsafe_call", f"禁止调用 {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.add(node, "dunder_access", f"禁止访问属性 {node.attr}")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, DENIED_NODES):
            self.add(node, "unsafe_syntax", f"禁止语法 {type(node).__name__}")
        super().generic_visit(node)


def validate_code(code: str) -> dict:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        issue = SafetyIssue(
            line=exc.lineno or 0,
            column=exc.offset or 0,
            code="syntax_error",
            message=exc.msg,
        )
        return {"accepted": False, "issues": [asdict(issue)]}
    visitor = SafetyVisitor()
    visitor.visit(tree)
    has_solid_assignment = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "solid"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        for node in ast.walk(tree)
    )
    if not has_solid_assignment:
        visitor.issues.append(
            SafetyIssue(0, 0, "missing_solid", "代码没有给最终变量 solid 赋值")
        )
    return {
        "accepted": not visitor.issues,
        "issues": [asdict(issue) for issue in visitor.issues],
    }
