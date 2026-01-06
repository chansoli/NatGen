import random
import re
from typing import List, Optional, Sequence, Tuple


CommentSpan = Tuple[int, int]
MutationCandidate = Tuple[int, int, str]


def find_comment_spans(code: str) -> List[CommentSpan]:
    """Locate line and block comment spans so mutations can avoid them."""
    spans: List[CommentSpan] = []
    i = 0
    n = len(code)
    in_line = False
    in_block = False
    in_str = False
    in_char = False
    start = 0
    escape = False

    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

        if in_line:
            if ch == "\n":
                spans.append((start, i))
                in_line = False
            i += 1
            continue

        if in_block:
            if ch == "*" and nxt == "/":
                spans.append((start, i + 2))
                in_block = False
                i += 2
                continue
            i += 1
            continue

        if in_str:
            if not escape and ch == "\\":
                escape = True
            elif escape:
                escape = False
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if in_char:
            if not escape and ch == "\\":
                escape = True
            elif escape:
                escape = False
            elif ch == "'":
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            start = i
            in_line = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            start = i
            in_block = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue
        i += 1

    if in_line:
        spans.append((start, n))
    if in_block:
        spans.append((start, n))
    return spans


def _filter_comment_candidates(candidates: Sequence[MutationCandidate], comment_spans: Sequence[CommentSpan]):
    if not comment_spans:
        return list(candidates)
    filtered = []
    for start, end, repl in candidates:
        in_comment = any(span_start <= start < span_end for span_start, span_end in comment_spans)
        if not in_comment:
            filtered.append((start, end, repl))
    return filtered


def find_relational_candidates(code: str) -> List[MutationCandidate]:
    candidates: List[MutationCandidate] = []
    for match in re.finditer(r"<=", code):
        candidates.append((match.start(), match.end(), "<"))
    for match in re.finditer(r">=", code):
        candidates.append((match.start(), match.end(), ">"))
    for match in re.finditer(r"(?<!<)<(?![=<])", code):
        candidates.append((match.start(), match.end(), "<="))
    for match in re.finditer(r"(?<!>)>(?![=>])", code):
        candidates.append((match.start(), match.end(), ">="))
    return candidates


def find_equality_candidates(code: str) -> List[MutationCandidate]:
    candidates: List[MutationCandidate] = []
    for match in re.finditer(r"==", code):
        candidates.append((match.start(), match.end(), "!="))
    for match in re.finditer(r"!=", code):
        candidates.append((match.start(), match.end(), "=="))
    return candidates


def find_constant_candidates(code: str) -> List[MutationCandidate]:
    candidates: List[MutationCandidate] = []
    for match in re.finditer(r"\b\d+\b", code):
        val = int(match.group(0))
        new_val = val + 1 if val == 0 else val - 1
        candidates.append((match.start(), match.end(), str(new_val)))
    return candidates


def find_arithmetic_candidates(code: str) -> List[MutationCandidate]:
    candidates: List[MutationCandidate] = []
    for match in re.finditer(r"\s[\+\-]\s", code):
        op = match.group(0).strip()
        repl = "-" if op == "+" else "+"
        start = match.start() + 1
        end = match.end() - 1
        candidates.append((start, end, repl))
    return candidates


def pick_arithmetic_candidate(code: str, candidates: Sequence[MutationCandidate], rng: random.Random) -> Optional[MutationCandidate]:
    if not candidates:
        return None
    non_loop = []
    for start, end, repl in candidates:
        line_start = code.rfind("\n", 0, start)
        line_end = code.find("\n", end)
        if line_start == -1:
            line_start = 0
        if line_end == -1:
            line_end = len(code)
        line = code[line_start:line_end]
        if "for" not in line:
            non_loop.append((start, end, repl))
    pool = non_loop if non_loop else list(candidates)
    return rng.choice(pool) if pool else None


def apply_mutation(code: str, start: int, end: int, repl: str) -> str:
    return code[:start] + repl + code[end:]


def mutate_code(code: str, rng: random.Random) -> Optional[str]:
    mutations = []
    comment_spans = find_comment_spans(code)
    rel = _filter_comment_candidates(find_relational_candidates(code), comment_spans)
    if rel:
        mutations.append(("relational", rel))
    eq = _filter_comment_candidates(find_equality_candidates(code), comment_spans)
    if eq:
        mutations.append(("equality", eq))
    consts = _filter_comment_candidates(find_constant_candidates(code), comment_spans)
    if consts:
        mutations.append(("constant", consts))
    arith = _filter_comment_candidates(find_arithmetic_candidates(code), comment_spans)
    if arith:
        mutations.append(("arithmetic", arith))
    if not mutations:
        return None
    mtype, candidates = rng.choice(mutations)
    if mtype == "arithmetic":
        choice = pick_arithmetic_candidate(code, candidates, rng)
        if choice is None:
            return None
        start, end, repl = choice
    else:
        start, end, repl = rng.choice(candidates)
    mutated = apply_mutation(code, start, end, repl)
    if mutated == code:
        return None
    return mutated


class SemanticBreakingTransformation:
    """Applies a single semantics-breaking mutation to code (deviant generator)."""

    def transform_code(self, code: str, rng: random.Random) -> Optional[str]:
        return mutate_code(code, rng)


__all__ = [
    "find_comment_spans",
    "find_relational_candidates",
    "find_equality_candidates",
    "find_constant_candidates",
    "find_arithmetic_candidates",
    "pick_arithmetic_candidate",
    "apply_mutation",
    "mutate_code",
    "SemanticBreakingTransformation",
]
