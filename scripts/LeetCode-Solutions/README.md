# LeetCode C++ Dataset Generator

This script builds triplet/pair datasets from a local LeetCode C++ solutions repo using the same variant/deviant pipelines as the POJ-104 flow.

## Prerequisites
- Python environment with project deps installed (activate the NatGen env).
- Tree-sitter parser built at `parser/languages.so` (run `./setup.sh` if needed).
- Sibling repo layout (default):
  - NatGen at `.../CloneSearchExp/NatGen`
  - LeetCode solutions at `.../CloneSearchExp/LeetCode-Solutions`

## Quick start (defaults)
From the NatGen repo root:

```bash
python scripts/LeetCode-Solutions/make_dataset.py
```

Defaults:
- dataset_root: `../LeetCode-Solutions` (recurses for `*.cpp`)
- output_root: `data/leetcode`
- max_files: 20
- train_frac: 0.8
- kpos/kneg: 1/1
- validate_candidates: on (with stub `main`), disable via `--no-validate_candidates`
- wrap_main: on, disable via `--no-wrap_main`

Outputs:
- Generated code under `data/leetcode/gen/{train,test}/`
- Datasets under `data/leetcode/datasets/` (`triplets_train.jsonl`, `triplets_valid.jsonl`, `pairs_train.jsonl`, `pairs_test.jsonl`, `metadata.jsonl`, `splits.json`)

## Common overrides
- Change dataset root:
  ```bash
  python scripts/LeetCode-Solutions/make_dataset.py --dataset_root /path/to/LeetCode-Cpp
  ```
- Disable compilation:
  ```bash
  python scripts/LeetCode-Solutions/make_dataset.py --no-validate_candidates --no-wrap_main
  ```
- Process more files:
  ```bash
  python scripts/LeetCode-Solutions/make_dataset.py --max_files 200
  ```

## Notes
- The script extracts the first `class Solution` in each file; files without one are logged in `metadata.jsonl`.
- Compilation is best-effort only; there is no oracle/harness.
- `tree_token_ids` are placeholder zeros (128) to match the POJ pipeline.
