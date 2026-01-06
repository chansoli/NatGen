import importlib.util
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "poj104" / "make_dataset.py"
_spec = importlib.util.spec_from_file_location("make_dataset_module", MODULE_PATH)
_make_dataset = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_make_dataset)


mutate_code = _make_dataset.mutate_code


def test_relational_mutation():
    rng = random.Random(0)
    code = "int main(){if(a<=b) return a; return b;}"
    mutated = mutate_code(code, rng)
    assert mutated is not None
    assert "<=" not in mutated
    assert "<" in mutated


def test_equality_mutation():
    rng = random.Random(1)
    code = "int main(){if(a==b) return a; else return b;}"
    mutated = mutate_code(code, rng)
    assert mutated is not None
    assert "==" not in mutated
    assert "!=" in mutated


def test_constant_mutation():
    rng = random.Random(2)
    code = "int main(){int x = 0; return x;}"
    mutated = mutate_code(code, rng)
    assert mutated is not None
    assert " 0;" not in mutated
    assert " 1;" in mutated


def test_arithmetic_mutation():
    rng = random.Random(3)
    code = "int main(){int c = a + b; return c;}"
    mutated = mutate_code(code, rng)
    assert mutated is not None
    assert " + " not in mutated
    assert " - " in mutated


def test_comment_only_returns_none():
    rng = random.Random(4)
    comment_line = "// if (a <= b) return a;"
    code = f"{comment_line}\nint main(){{}}"
    mutated = mutate_code(code, rng)
    assert mutated is None


def test_comment_not_mutated_when_real_candidate_exists():
    rng = random.Random(5)
    comment_line = "// compare a <= b"
    code = f"{comment_line}\nint main(){{int a=1,b=2; if(a<=b) return a; return b;}}"
    mutated = mutate_code(code, rng)
    assert mutated is not None
    assert mutated.splitlines()[0] == comment_line
    assert mutated != code
