import ast
from typing import Iterable, Set, Dict, List, Tuple


def _node_src(source: str, node: ast.AST) -> str:
    """Return best-effort original source for a node."""
    seg = ast.get_source_segment(source, node)
    if seg:
        return seg
    try:
        return ast.unparse(node)  # Python 3.9+
    except Exception:
        import astor
        return astor.to_source(node).rstrip()


def _top_level_defs(code: str) -> Dict[str, Set[str]]:
    """
    Parse code and return a map:
      kind:name -> set of normalized AST dumps (one per duplicate name).
    kind is 'func' or 'class'.
    """
    tree = ast.parse(code)
    seen: Dict[str, Set[str]] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = f"func:{node.name}"
            dump = ast.dump(node, include_attributes=False)
            seen.setdefault(key, set()).add(dump)
        elif isinstance(node, ast.ClassDef):
            key = f"class:{node.name}"
            dump = ast.dump(node, include_attributes=False)
            seen.setdefault(key, set()).add(dump)
    return seen


def _node_span(node: ast.AST) -> Tuple[int, int]:
    """
    Get (start_line, end_line) for a top-level function/class node,
    including decorators. Lines are 1-based, inclusive.
    """
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)

    # Include decorators if present
    decos = getattr(node, "decorator_list", None)
    if decos:
        dmin = min(getattr(d, "lineno", start) for d in decos)
        if start is None or (dmin is not None and dmin < start):
            start = dmin

    if start is None or end is None:
        raise ValueError("Python AST lacks lineno/end_lineno; need Python 3.8+.")
    return start, end


def remove_duplicate_defs(src_a: str, src_b: str) -> Tuple[str, List[str]]:
    """
    Remove duplicate *top-level* function/class definitions from src_b
    that are structurally identical to definitions present in src_a.

    Returns:
      (new_src_b, removed_blocks)
        - new_src_b: modified source B string
        - removed_blocks: list of removed code blocks (as strings)

    Notes:
      - Comparison is structural (AST), not textual.
      - Only top-level defs are considered (no nested functions/classes).
      - Formatting of remaining code in src_b is preserved.
    """
    a_defs = _top_level_defs(src_a)

    tree_b = ast.parse(src_b)
    lines = src_b.splitlines(keepends=True)

    # Collect spans to remove and also capture the exact text for reporting
    spans_to_remove: List[Tuple[int, int]] = []
    removed_blocks: List[str] = []

    for node in tree_b.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = f"func:{node.name}"
            dump = ast.dump(node, include_attributes=False)
            if key in a_defs and dump in a_defs[key]:
                s, e = _node_span(node)
                spans_to_remove.append((s, e))
        elif isinstance(node, ast.ClassDef):
            key = f"class:{node.name}"
            dump = ast.dump(node, include_attributes=False)
            if key in a_defs and dump in a_defs[key]:
                s, e = _node_span(node)
                spans_to_remove.append((s, e))

    # Merge overlapping/adjacent spans (just in case)
    spans_to_remove.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans_to_remove:
        if not merged or s > merged[-1][1] + 0:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    # Build output while capturing removed text
    out_lines: List[str] = []
    cur = 1  # 1-based line counter
    for s, e in merged:
        # keep lines before span
        if cur <= s - 1:
            out_lines.extend(lines[cur-1:s-1])
        # capture removed block
        removed_blocks.append("".join(lines[s-1:e]))
        # skip span
        cur = e + 1
    # keep remaining lines
    if cur <= len(lines):
        out_lines.extend(lines[cur-1:])

    return "".join(out_lines) #, removed_blocks

def split_imports_and_code(code: str):
    tree = ast.parse(code)

    # Collect line numbers of all import nodes
    import_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for i in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                import_lines.add(i)

    imports = []
    rest = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        if lineno in import_lines:
            imports.append(line)
        else:
            rest.append(line)

    return "\n".join(imports), "\n".join(rest)
