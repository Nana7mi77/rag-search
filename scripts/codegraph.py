#!/usr/bin/env python3
"""
codegraph - 轻量级Python代码关系分析工具
用于快速了解项目模块关系、调用链、候选落点
"""

import ast
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


class CodeGraph:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = {}
        self.imports = defaultdict(list)
        self.classes = defaultdict(list)
        self.functions = defaultdict(list)
        self.call_graph = defaultdict(set)
        self.reverse_imports = defaultdict(list)

    def parse_file(self, filepath):
        rel = filepath.relative_to(self.root_dir)
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError) as e:
            return {"file": str(rel), "error": str(e)}

        info = {
            "file": str(rel),
            "imports": [],
            "from_imports": [],
            "classes": [],
            "functions": [],
            "calls": [],
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    info["imports"].append(alias.name)
                    self.imports[str(rel)].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full = f"{module}.{alias.name}" if module else alias.name
                    info["from_imports"].append(full)
                    self.imports[str(rel)].append(module)
                    if module:
                        self.reverse_imports[module].append(str(rel))
            elif isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(ast.dump(base))
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                info["classes"].append({
                    "name": node.name,
                    "bases": bases,
                    "methods": methods,
                    "line": node.lineno,
                })
                self.classes[str(rel)].append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not any(
                    isinstance(p, ast.ClassDef)
                    for p in ast.walk(tree)
                    if node in ast.walk(p)
                ):
                    info["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [a.arg for a in node.args.args],
                    })
                    self.functions[str(rel)].append(node.name)
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name:
                    info["calls"].append(func_name)
                    self.call_graph[str(rel)].add(func_name)

        return info

    def scan(self):
        results = []
        for py_file in sorted(self.root_dir.rglob("*.py")):
            if any(
                skip in str(py_file)
                for skip in ["__pycache__", ".git", "node_modules", "venv", ".venv"]
            ):
                continue
            info = self.parse_file(py_file)
            results.append(info)
        self.files = {r["file"]: r for r in results}
        return results

    def find_module_references(self, module_name):
        refs = []
        for file, imports in self.imports.items():
            for imp in imports:
                if module_name in imp:
                    refs.append(file)
        return list(set(refs))

    def find_symbol_usage(self, symbol):
        usage = []
        for file, info in self.files.items():
            for call in info.get("calls", []):
                if call == symbol:
                    usage.append(file)
            for cls in info.get("classes", []):
                for method in cls.get("methods", []):
                    if method == symbol:
                        usage.append(file)
        return list(set(usage))

    def get_dependency_graph(self):
        graph = {}
        for file, imports in self.imports.items():
            local_deps = []
            for imp in imports:
                if imp and not imp.startswith((".", "_")):
                    continue
                for other_file in self.files:
                    if imp and imp in other_file:
                        local_deps.append(other_file)
            if local_deps:
                graph[file] = list(set(local_deps))
        return graph

    def generate_report(self):
        self.scan()
        report = {
            "summary": {
                "total_files": len(self.files),
                "total_classes": sum(len(v) for v in self.classes.values()),
                "total_functions": sum(len(v) for v in self.functions.values()),
            },
            "modules": {},
            "dependency_graph": self.get_dependency_graph(),
            "reverse_dependencies": dict(self.reverse_imports),
        }

        for file, info in self.files.items():
            report["modules"][file] = {
                "imports": info.get("imports", []),
                "from_imports": info.get("from_imports", []),
                "classes": [c["name"] for c in info.get("classes", [])],
                "functions": [f["name"] for f in info.get("functions", [])],
                "calls": list(info.get("calls", [])),
            }

        return report

    def print_summary(self):
        report = self.generate_report()
        s = report["summary"]
        print(f"Files: {s['total_files']}, Classes: {s['total_classes']}, Functions: {s['total_functions']}")
        print("\n--- Module Overview ---")
        for file, mod in sorted(report["modules"].items()):
            classes = mod["classes"]
            funcs = mod["functions"]
            imports = [i for i in mod["imports"] if i]
            print(f"\n  {file}")
            if classes:
                print(f"    classes: {', '.join(classes)}")
            if funcs:
                print(f"    functions: {', '.join(funcs[:10])}")
            if imports:
                local = [i for i in imports if i.startswith("rag_search")]
                if local:
                    print(f"    internal deps: {', '.join(local)}")

        print("\n--- Dependency Graph (internal) ---")
        for file, deps in sorted(report["dependency_graph"].items()):
            if deps:
                print(f"  {file} -> {', '.join(deps)}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    output = sys.argv[2] if len(sys.argv) > 2 else None

    graph = CodeGraph(root)
    report = graph.generate_report()

    if output:
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Report written to {output}")
    else:
        graph.print_summary()


if __name__ == "__main__":
    main()
