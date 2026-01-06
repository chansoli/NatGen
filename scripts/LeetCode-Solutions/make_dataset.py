import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
# Walk up until we find the repo root that contains src/
REPO_ROOT = SCRIPT_DIR
for _ in range(4):
    if (REPO_ROOT / "src").is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_preprocessors.semantic_breaking_transformation import (
    SemanticBreakingTransformation,
)
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


def list_cpp_files(dataset_root: Path):
    return sorted(p for p in dataset_root.rglob("*.cpp") if p.is_file())


def split_files(files, train_frac: float, seed: int):
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    split = int(len(files) * train_frac)
    return files[:split], files[split:]


def read_code(path: Path) -> str:
    return path.read_text(errors="ignore")


def extract_first_solution(code: str) -> Tuple[Optional[str], str]:
    match = re.search(r"\bclass\s+Solution\b", code)
    if not match:
        return None, "no_solution_class"
    brace_start = code.find("{", match.end())
    if brace_start == -1:
        return None, "no_open_brace"
    depth = 0
    end_idx = None
    for i in range(brace_start, len(code)):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < len(code) and code[j] in " \t\r\n":
                    j += 1
                if j < len(code) and code[j] == ";":
                    j += 1
                end_idx = j
                break
    if end_idx is None:
        return None, "unclosed_class"

    # Truncate before any subsequent Solution class to honor "choose first" rule.
    remainder = code[end_idx:]
    next_match = re.search(r"\bclass\s+Solution\b", remainder)
    tail_end = end_idx + next_match.start() if next_match else len(code)

    extracted = code[:tail_end]
    return extracted, "ok"


def wrap_code(code: str, wrap_main: bool) -> str:
    main_stub = "\nint main(){return 0;}\n" if wrap_main else ""
    return "#include <bits/stdc++.h>\nusing namespace std;\n" + code + main_stub


def compile_cpp(code: str, work_dir: Path, name: str, wrap_main: bool) -> Optional[Path]:
    src_path = work_dir / f"{name}.cpp"
    bin_path = work_dir / f"{name}.out"
    src_path.write_text(wrap_code(code, wrap_main))
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
    # default: sibling repo LeetCode-Solutions next to NatGen
    parser.add_argument("--dataset_root", type=str, default="../LeetCode-Solutions")
    parser.add_argument("--output_root", type=str, default="data/leetcode")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--kpos", type=int, default=1)
    parser.add_argument("--kneg", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate_candidates", dest="validate_candidates", action="store_true", help="Compile candidates (no oracle) with optional stub main")
    parser.add_argument("--no-validate_candidates", dest="validate_candidates", action="store_false", help="Skip compilation of candidates")
    parser.add_argument("--wrap_main", dest="wrap_main", action="store_true", help="Append an empty main() when compiling candidates")
    parser.add_argument("--no-wrap_main", dest="wrap_main", action="store_false", help="Do not append stub main() when compiling candidates")
    parser.set_defaults(validate_candidates=True, wrap_main=True)
    parser.add_argument("--reuse_splits", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = (REPO_ROOT / dataset_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

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

    files = list_cpp_files(dataset_root)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print("No .cpp files found; exiting.")
        return

    if args.reuse_splits and splits_path.exists():
        splits = json.loads(splits_path.read_text())
        train_files = [Path(p) for p in splits["train"]]
        test_files = [Path(p) for p in splits["test"]]
    else:
        train_files, test_files = split_files(files, args.train_frac, args.seed)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps({"train": [str(p) for p in train_files], "test": [str(p) for p in test_files]}, indent=2))

    variant_gen = VariantGenerator(parser_path, config_file)
    deviant_gen = SemanticBreakingTransformation()

    triplets = []
    pairs_train = []
    pairs_test = []
    metadata_rows = []

    rng = random.Random(args.seed)

    for split_name, file_list in [("train", train_files), ("test", test_files)]:
        for file_path in file_list:
            code_raw = read_code(file_path)
            extracted, reason = extract_first_solution(code_raw)
            if extracted is None:
                metadata_rows.append({
                    "file": str(file_path),
                    "problem_id": file_path.stem,
                    "orig_id": file_path.stem,
                    "compiler_ok": None,
                    "parse_reason": reason,
                    "reason": "extract_fail",
                })
                continue

            problem_id = file_path.stem
            orig_id = file_path.stem
            problem_out_dir = gen_root / split_name / problem_id
            problem_out_dir.mkdir(parents=True, exist_ok=True)
            out_orig_path = problem_out_dir / f"{orig_id}.cpp"
            out_orig_path.write_text(extracted)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                if args.validate_candidates:
                    orig_bin = compile_cpp(extracted, tmpdir_path, "orig", args.wrap_main)
                    metadata_rows.append({
                        "file": str(file_path),
                        "problem_id": problem_id,
                        "orig_id": orig_id,
                        "compiler_ok": orig_bin is not None,
                        "parse_reason": reason,
                        "reason": None if orig_bin is not None else "orig_compile_fail",
                    })
                    if orig_bin is None:
                        continue
                else:
                    orig_bin = None
                    metadata_rows.append({
                        "file": str(file_path),
                        "problem_id": problem_id,
                        "orig_id": orig_id,
                        "compiler_ok": None,
                        "parse_reason": reason,
                        "reason": "validation_skipped_orig",
                    })

                variants = []
                attempts = 0
                while len(variants) < args.kpos and attempts < args.kpos * 10:
                    attempts += 1
                    cand_code = variant_gen.generate(extracted)
                    if cand_code is None:
                        continue
                    if args.validate_candidates:
                        cand_bin = compile_cpp(cand_code, tmpdir_path, f"var_{attempts}", args.wrap_main)
                        metadata_rows.append({
                            "file": str(file_path),
                            "problem_id": problem_id,
                            "orig_id": orig_id,
                            "variant_id": attempts,
                            "compiler_ok": cand_bin is not None,
                            "parse_reason": reason,
                            "reason": None if cand_bin is not None else "variant_compile_fail",
                        })
                        if cand_bin is None:
                            continue
                    else:
                        metadata_rows.append({
                            "file": str(file_path),
                            "problem_id": problem_id,
                            "orig_id": orig_id,
                            "variant_id": attempts,
                            "compiler_ok": None,
                            "parse_reason": reason,
                            "reason": "validation_skipped",
                        })
                    var_id = len(variants)
                    out_var_path = problem_out_dir / f"{orig_id}_v{var_id}.cpp"
                    out_var_path.write_text(cand_code)
                    variants.append(cand_code)

                deviants = []
                attempts = 0
                while len(deviants) < args.kneg and attempts < args.kneg * 20:
                    attempts += 1
                    cand_code = deviant_gen.transform_code(extracted, rng)
                    if cand_code is None:
                        continue
                    if args.validate_candidates:
                        cand_bin = compile_cpp(cand_code, tmpdir_path, f"dev_{attempts}", args.wrap_main)
                        metadata_rows.append({
                            "file": str(file_path),
                            "problem_id": problem_id,
                            "orig_id": orig_id,
                            "deviant_id": attempts,
                            "compiler_ok": cand_bin is not None,
                            "parse_reason": reason,
                            "reason": None if cand_bin is not None else "deviant_compile_fail",
                        })
                        if cand_bin is None:
                            continue
                    else:
                        metadata_rows.append({
                            "file": str(file_path),
                            "problem_id": problem_id,
                            "orig_id": orig_id,
                            "deviant_id": attempts,
                            "compiler_ok": None,
                            "parse_reason": reason,
                            "reason": "validation_skipped",
                        })
                    dev_id = len(deviants)
                    out_dev_path = problem_out_dir / f"{orig_id}_d{dev_id}.cpp"
                    out_dev_path.write_text(cand_code)
                    deviants.append(cand_code)

            if len(variants) < args.kpos or len(deviants) < args.kneg:
                continue

            if split_name == "train":
                triplet_count = min(args.kpos, args.kneg)
                for i in range(triplet_count):
                    triplets.append({
                        "orig_code": extracted,
                        "positive_code": variants[i],
                        "negative_code": deviants[i],
                        "tree_token_ids": ZERO_TREE_TOKEN_IDS,
                    })

            pair_list = pairs_train if split_name == "train" else pairs_test
            for var_code in variants:
                pair_list.append({
                    "orig_code": extracted,
                    "cand_code": var_code,
                    "label": 1,
                })
            for dev_code in deviants:
                pair_list.append({
                    "orig_code": extracted,
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


if __name__ == "__main__":
    main()
