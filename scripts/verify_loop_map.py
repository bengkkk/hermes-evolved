#!/usr/bin/env python3
"""AST line-number verification for the think_daemon_loop.md navigation maps.

Prints every top-level function/class and method def with its line number
for think_daemon.py and world_model.py at the current HEAD, so the docs'
navigation-map tables can be re-verified after each code change.
"""
import ast
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Exit cleanly when piped into head/less (BrokenPipeError -> SIGPIPE default)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def dump(path: Path) -> None:
    tree = ast.parse(path.read_text())
    print(f"=== {path.name} ===")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"{node.lineno:5d}  {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}")
        elif isinstance(node, ast.ClassDef):
            print(f"{node.lineno:5d}  class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    print(f"{sub.lineno:5d}      {'async ' if isinstance(sub, ast.AsyncFunctionDef) else ''}def {sub.name}")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["think_daemon.py", "world_model.py"]:
        dump(ROOT / name)
        print()
