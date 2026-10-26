"""
scripts.hygiene_check
AST static hygiene checker enforcing:
1. Explicit __all__ in every production module (including __init__.py).
2. Zero mutable default arguments in sync and async defs.
3. DG-33 / Commandment 2: Zero wall-clock calls outside red/_time.py.
"""

import ast
import sys
from pathlib import Path

__all__ = ["check_file", "main"]

_IMMUTABLE_CALL_ALLOWLIST = {
    "tuple",
    "frozenset",
    "bytes",
    "str",
    "int",
    "float",
    "bool",
    "complex",
    "Decimal",
    "Path",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
}
_TIME_FUNCS = {"time", "time_ns"}
_DATETIME_FUNCS = {"now", "utcnow"}


def check_file(filepath: Path, repo_root: Path) -> list[str]:
    violations: list[str] = []
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except Exception as exc:
        return [f"Failed to parse: {exc}"]

    try:
        rel_path = filepath.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel_path = filepath.as_posix()

    is_time_module = rel_path == "red/_time.py"
    is_test = rel_path.startswith("tests/")

    has_all = False
    time_module_aliases: set[str] = set()
    direct_time_funcs: set[str] = set()
    datetime_module_aliases: set[str] = set()
    datetime_class_aliases: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    has_all = True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                has_all = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                if alias.name == "time":
                    time_module_aliases.add(bound_name)
                elif alias.name == "datetime":
                    datetime_module_aliases.add(bound_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "time":
                for alias in node.names:
                    if alias.name in _TIME_FUNCS:
                        direct_time_funcs.add(alias.asname or alias.name)
            elif node.module == "datetime":
                for alias in node.names:
                    if alias.name == "datetime":
                        datetime_class_aliases.add(alias.asname or alias.name)
                    elif alias.name in _DATETIME_FUNCS:
                        direct_time_funcs.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in list(node.args.defaults) + list(node.args.kw_defaults):
                if default is None:
                    continue
                if isinstance(default, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
                    violations.append(
                        f"Line {default.lineno}: Function '{node.name}' uses a mutable literal/comprehension default."
                    )
                elif isinstance(default, ast.Call):
                    func_name = ""
                    if isinstance(default.func, ast.Name):
                        func_name = default.func.id
                    elif isinstance(default.func, ast.Attribute):
                        func_name = default.func.attr
                    if func_name not in _IMMUTABLE_CALL_ALLOWLIST:
                        violations.append(
                            f"Line {default.lineno}: Function '{node.name}' uses a non-allowlisted call default '{func_name or '<expr>'}()'."
                        )

        if isinstance(node, ast.Call) and not is_time_module:
            if isinstance(node.func, ast.Name) and node.func.id in direct_time_funcs:
                violations.append(
                    f"Line {node.lineno}: DG-33 Violation (direct call to '{node.func.id}()' outside red/_time.py)."
                )
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                val = node.func.value
                if isinstance(val, ast.Name) and val.id in time_module_aliases and attr in _TIME_FUNCS:
                    violations.append(
                        f"Line {node.lineno}: DG-33 Violation ('{val.id}.{attr}()' outside red/_time.py)."
                    )
                elif isinstance(val, ast.Name) and val.id in datetime_class_aliases and attr in _DATETIME_FUNCS:
                    violations.append(
                        f"Line {node.lineno}: DG-33 Violation ('{val.id}.{attr}()' outside red/_time.py)."
                    )
                elif (
                    isinstance(val, ast.Attribute)
                    and isinstance(val.value, ast.Name)
                    and val.value.id in datetime_module_aliases
                    and val.attr == "datetime"
                    and attr in _DATETIME_FUNCS
                ):
                    violations.append(
                        f"Line {node.lineno}: DG-33 Violation ('{val.value.id}.datetime.{attr}()' outside red/_time.py)."
                    )

    if not has_all and not is_test:
        violations.append("Missing explicit __all__ declaration.")

    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    red_dir = repo_root / "red"
    tests_dir = repo_root / "tests"

    files_to_check: list[Path] = []
    if red_dir.exists():
        files_to_check.extend(sorted(red_dir.rglob("*.py")))
    if tests_dir.exists():
        files_to_check.extend(sorted(tests_dir.rglob("*.py")))

    all_clean = True
    for py_file in files_to_check:
        issues = check_file(py_file, repo_root)
        if issues:
            all_clean = False
            for issue in issues:
                print(f"{py_file.relative_to(repo_root)}: {issue}")

    return 0 if all_clean else 1


if __name__ == "__main__":
    sys.exit(main())