import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from typing import Optional
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from input_generators import get_input_generator
from src.data_preprocessors.transformations import SemanticPreservingTransformation
from src.pretraining.prepare_data import create_transformers_from_conf_file


ZERO_TREE_TOKEN_IDS = " ".join(["0"] * 128)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    except Exception:
        pass


def list_problem_ids(dataset_root: Path):
    return sorted(int(p.name) for p in dataset_root.iterdir() if p.is_dir() and p.name.isdigit())


def choose_splits(problem_ids, total, train_count, seed):
    rng = random.Random(seed)
    ids = problem_ids[:]
    rng.shuffle(ids)
    selected = ids[:total]
    train = selected[:train_count]
    test = selected[train_count:]
    return train, test


def read_code(path: Path) -> str:
    return path.read_text(errors="ignore")


def wrap_code(code: str) -> str:
    return "#include <bits/stdc++.h>\nusing namespace std;\n" + code


def compile_cpp(code: str, work_dir: Path, name: str) -> Optional[Path]:
    src_path = work_dir / f"{name}.cpp"
    bin_path = work_dir / f"{name}.out"
    src_path.write_text(wrap_code(code))
    result = subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++17",
            "-w",
            "-fpermissive",
            str(src_path),
            "-o",
            str(bin_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None
    return bin_path


def run_binary(bin_path: Path, input_str: str, timeout_s: float):
    try:
        result = subprocess.run(
            [str(bin_path)],
            input=input_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if result.returncode != 0:
        return None, "runtime_error"
    return result.stdout.strip(), None


def differential_oracle(orig_bin: Path, cand_bin: Path, inputs, timeout_s: float):
    for test_input in inputs:
        orig_out, orig_err = run_binary(orig_bin, test_input, timeout_s)
        if orig_err is not None:
            return False, "orig_fail"
        cand_out, cand_err = run_binary(cand_bin, test_input, timeout_s)
        if cand_err is not None:
            return True, "cand_fail"
        if orig_out != cand_out:
            return True, "diff"
    return False, "match"


def find_comment_spans(code: str):
    spans = []
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
            elif ch == "\"":
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
        if ch == "\"":
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


def _filter_comment_candidates(candidates, comment_spans):
    if not comment_spans:
        return candidates
    filtered = []
    for start, end, repl in candidates:
        in_comment = any(span_start <= start < span_end for span_start, span_end in comment_spans)
        if not in_comment:
            filtered.append((start, end, repl))
    return filtered


def find_relational_candidates(code: str):
    candidates = []
    for match in re.finditer(r"<=", code):
        candidates.append((match.start(), match.end(), "<"))
    for match in re.finditer(r">=", code):
        candidates.append((match.start(), match.end(), ">"))
    for match in re.finditer(r"(?<!<)<(?![=<])", code):
        candidates.append((match.start(), match.end(), "<="))
    for match in re.finditer(r"(?<!>)>(?![=>])", code):
        candidates.append((match.start(), match.end(), ">="))
    return candidates


def find_equality_candidates(code: str):
    candidates = []
    for match in re.finditer(r"==", code):
        candidates.append((match.start(), match.end(), "!="))
    for match in re.finditer(r"!=", code):
        candidates.append((match.start(), match.end(), "=="))
    return candidates


def find_constant_candidates(code: str):
    candidates = []
    for match in re.finditer(r"\b\d+\b", code):
        val = int(match.group(0))
        new_val = val + 1 if val == 0 else val - 1
        candidates.append((match.start(), match.end(), str(new_val)))
    return candidates


def find_arithmetic_candidates(code: str):
    candidates = []
    for match in re.finditer(r"\s[\+\-]\s", code):
        op = match.group(0).strip()
        if op == "+":
            repl = "-"
        else:
            repl = "+"
        start = match.start() + 1
        end = match.end() - 1
        candidates.append((start, end, repl))
    return candidates


def pick_arithmetic_candidate(code: str, candidates, rng: random.Random):
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
    pool = non_loop if non_loop else candidates
    return rng.choice(pool)


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


class VariantGenerator:
    def __init__(self, parser_path: Path, config_file: Path):
        config = json.loads(config_file.read_text())
        transformers = create_transformers_from_conf_file(config["transformers"])
        self.transformer = SemanticPreservingTransformation(
            parser_path=str(parser_path),
            language="cpp",
            transform_functions=transformers,
        )

    def generate(self, code: str):
        transformed_code, used_transformer = self.transformer.transform_code(code)
        if not used_transformer:
            return None
        return transformed_code


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="dataset/ProgramData")
    parser.add_argument("--output_root", type=str, default="data")
    parser.add_argument("--problems", type=int, default=20)
    parser.add_argument("--train_problems", type=int, default=16)
    parser.add_argument("--solutions_per_problem", type=int, default=10)
    parser.add_argument("--kpos", type=int, default=1)
    parser.add_argument("--kneg", type=int, default=1)
    parser.add_argument("--tests_per_original", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout_s", type=float, default=1.5)
    parser.add_argument("--validate_candidates", action="store_true", help="Compile and run oracle tests for variants/deviants")
    parser.add_argument("--reuse_splits", action="store_true")
    parser.add_argument("--pilot_only", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    gen_root = output_root / "gen"
    datasets_root = output_root / "datasets"
    splits_path = datasets_root / "splits.json"
    parser_path = REPO_ROOT / "parser" / "languages.so"
    config_file = REPO_ROOT / "configs" / "pretraining" / "data_config" / "data_processing_config.json"

    print("Working directories:")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_root: {output_root}")
    print(f"  gen_root: {gen_root}")
    print(f"  datasets_root: {datasets_root}")

    problem_ids = list_problem_ids(dataset_root)
    if args.reuse_splits and splits_path.exists():
        splits = json.loads(splits_path.read_text())
        train_ids = splits["train"]
        test_ids = splits["test"]
    else:
        train_ids, test_ids = choose_splits(problem_ids, args.problems, args.train_problems, args.seed)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps({"train": train_ids, "test": test_ids}, indent=2))

    if args.pilot_only:
        pilot_ids = [pid for pid in train_ids + test_ids if pid in {10, 16, 43, 66}]
        train_ids = [pid for pid in train_ids if pid in pilot_ids]
        test_ids = [pid for pid in test_ids if pid in pilot_ids]

    variant_gen = VariantGenerator(parser_path, config_file)

    triplets = []
    pairs_train = []
    pairs_test = []
    metadata_rows = []

    rng = random.Random(args.seed)

    for split_name, problem_list in [("train", train_ids), ("test", test_ids)]:
        for problem_id in problem_list:
            problem_dir = dataset_root / str(problem_id)
            if not problem_dir.exists():
                continue
            try:
                input_gen = get_input_generator(problem_id)
            except NotImplementedError:
                continue
            files = sorted(problem_dir.glob("*.txt"))
            rng.shuffle(files)
            selected = files[: args.solutions_per_problem]
            for file_path in selected:
                orig_code = read_code(file_path)
                orig_id = file_path.stem

                problem_out_dir = gen_root / split_name / str(problem_id)
                problem_out_dir.mkdir(parents=True, exist_ok=True)
                out_orig_path = problem_out_dir / f"{orig_id}.cpp"
                out_orig_path.write_text(orig_code)

                tests = [input_gen(rng) for _ in range(args.tests_per_original)]

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    if args.validate_candidates:
                        orig_bin = compile_cpp(orig_code, tmpdir_path, "orig")
                        if orig_bin is None:
                            metadata_rows.append({
                                "problem_id": problem_id,
                                "orig_id": orig_id,
                                "compiler_ok": False,
                                "oracle_tests": args.tests_per_original,
                                "diff_found": None,
                                "reason": "orig_compile_fail",
                            })
                            continue
                    else:
                        # Validation disabled; treat original as valid without compiling.
                        orig_bin = None
                        metadata_rows.append({
                            "problem_id": problem_id,
                            "orig_id": orig_id,
                            "compiler_ok": None,
                            "oracle_tests": 0,
                            "diff_found": None,
                            "reason": "validation_skipped_orig",
                        })

                    # Variants
                    variants = []
                    attempts = 0
                    while len(variants) < args.kpos and attempts < args.kpos * 10:
                        attempts += 1
                        cand_code = variant_gen.generate(orig_code)
                        if cand_code is None:
                            continue
                        if args.validate_candidates:
                            cand_bin = compile_cpp(cand_code, tmpdir_path, f"var_{attempts}")
                            if cand_bin is None:
                                metadata_rows.append({
                                    "problem_id": problem_id,
                                    "orig_id": orig_id,
                                    "variant_id": attempts,
                                    "compiler_ok": False,
                                    "oracle_tests": args.tests_per_original,
                                    "diff_found": None,
                                    "reason": "variant_compile_fail",
                                })
                                continue
                            diff_found, reason = differential_oracle(orig_bin, cand_bin, tests, args.timeout_s)
                            metadata_rows.append({
                                "problem_id": problem_id,
                                "orig_id": orig_id,
                                "variant_id": attempts,
                                "compiler_ok": True,
                                "oracle_tests": args.tests_per_original,
                                "diff_found": diff_found,
                                "reason": reason,
                            })
                            if diff_found:
                                continue
                        else:
                            # Validation disabled; record placeholder metadata.
                            metadata_rows.append({
                                "problem_id": problem_id,
                                "orig_id": orig_id,
                                "variant_id": attempts,
                                "compiler_ok": None,
                                "oracle_tests": 0,
                                "diff_found": None,
                                "reason": "validation_skipped",
                            })
                        var_id = len(variants)
                        out_var_path = problem_out_dir / f"{orig_id}_v{var_id}.cpp"
                        out_var_path.write_text(cand_code)
                        variants.append(cand_code)

                    # Deviants
                    deviants = []
                    attempts = 0
                    while len(deviants) < args.kneg and attempts < args.kneg * 20:
                        attempts += 1
                        cand_code = mutate_code(orig_code, rng)
                        if cand_code is None:
                            continue
                        if args.validate_candidates:
                            cand_bin = compile_cpp(cand_code, tmpdir_path, f"dev_{attempts}")
                            if cand_bin is None:
                                metadata_rows.append({
                                    "problem_id": problem_id,
                                    "orig_id": orig_id,
                                    "deviant_id": attempts,
                                    "compiler_ok": False,
                                    "oracle_tests": args.tests_per_original,
                                    "diff_found": None,
                                    "reason": "deviant_compile_fail",
                                })
                                continue
                            diff_found, reason = differential_oracle(orig_bin, cand_bin, tests, args.timeout_s)
                            metadata_rows.append({
                                "problem_id": problem_id,
                                "orig_id": orig_id,
                                "deviant_id": attempts,
                                "compiler_ok": True,
                                "oracle_tests": args.tests_per_original,
                                "diff_found": diff_found,
                                "reason": reason,
                            })
                            if not diff_found:
                                continue
                        else:
                            # Validation disabled; record placeholder metadata.
                            metadata_rows.append({
                                "problem_id": problem_id,
                                "orig_id": orig_id,
                                "deviant_id": attempts,
                                "compiler_ok": None,
                                "oracle_tests": 0,
                                "diff_found": None,
                                "reason": "validation_skipped",
                            })
                        dev_id = len(deviants)
                        out_dev_path = problem_out_dir / f"{orig_id}_d{dev_id}.cpp"
                        out_dev_path.write_text(cand_code)
                        deviants.append(cand_code)

                    if len(variants) < args.kpos or len(deviants) < args.kneg:
                        continue

                    if split_name == "train":
                        # Produce as many triplets as requested by kpos/kneg (default 1) in order.
                        triplet_count = min(args.kpos, args.kneg)
                        for i in range(triplet_count):
                            triplets.append({
                                "orig_code": orig_code,
                                "positive_code": variants[i],
                                "negative_code": deviants[i],
                                "tree_token_ids": ZERO_TREE_TOKEN_IDS,
                            })

                    pair_list = pairs_train if split_name == "train" else pairs_test
                    for var_code in variants:
                        pair_list.append({
                            "orig_code": orig_code,
                            "cand_code": var_code,
                            "label": 1,
                        })
                    for dev_code in deviants:
                        pair_list.append({
                            "orig_code": orig_code,
                            "cand_code": dev_code,
                            "label": 0,
                        })

    write_jsonl(datasets_root / "triplets_train.jsonl", triplets)
    write_jsonl(datasets_root / "triplets_valid.jsonl", triplets)
    write_jsonl(datasets_root / "pairs_train.jsonl", pairs_train)
    write_jsonl(datasets_root / "pairs_test.jsonl", pairs_test)
    write_jsonl(datasets_root / "metadata.jsonl", metadata_rows)

    print("Output directories:")
    print(f"  datasets: {datasets_root}")
    print(f"  generated code: {gen_root}")
    print("Stats:")
    print(f"  triplets: {len(triplets)}")
    print(f"  pairs_train: {len(pairs_train)}")
    print(f"  pairs_test: {len(pairs_test)}")
    print(f"  metadata rows: {len(metadata_rows)}")

    failure_reasons = {
        "orig_compile_fail",
        "variant_compile_fail",
        "deviant_compile_fail",
        "orig_fail",
        "cand_fail",
        "diff",
    }
    failed_rows = [r for r in metadata_rows if r.get("reason") in failure_reasons or r.get("compiler_ok") is False]
    if failed_rows:
        print("Failed entries:")
        for r in failed_rows:
            kind = "orig"
            if "variant_id" in r:
                kind = f"variant#{r['variant_id']}"
            elif "deviant_id" in r:
                kind = f"deviant#{r['deviant_id']}"
            print(f"  problem {r['problem_id']} orig {r['orig_id']} ({kind}): reason={r.get('reason')} diff_found={r.get('diff_found')} compiler_ok={r.get('compiler_ok')}")
    else:
        print("Failed entries: none")


if __name__ == "__main__":
    main()
