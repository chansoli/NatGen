You are Copilot Agent building a LeetCode C++ dataset generator analogous to the POJ-104 pipeline.

GOAL
- Produce triplet and pair JSONL datasets from the LeetCode C++ solutions folder (externally mounted; path passed via CLI flag).
- Reuse the POJ pipeline structure while handling LeetCode-specific quirks (multiple Solution classes per file, no per-problem input generators, LeetCode-style function signatures).

REFERENCES
- Template script: [scripts/poj104/make_dataset.py](scripts/poj104/make_dataset.py)
- Existing plan for POJ data: [tasks/task-make-dataset-poj104cc.md](tasks/task-make-dataset-poj104cc.md)

ASSUMPTIONS
- Files are standalone .cpp solution submissions, often beginning with comments and omitting includes; many define only `class Solution` (LeetCode harness provides main/structs).
- Some files contain multiple `class Solution` or alternative solution classes; choose the first `class Solution` definition as the canonical body.
- We do not have per-problem input generators; compilation/oracle validation is stubbed and on by default (with `--wrap_main`), and can be disabled via `--no-validate_candidates` / `--no-wrap_main`.

DELIVERABLES
1) New script under scripts/LeetCode-Solutions/make_dataset.py that:
   - Enumerates .cpp files under a configurable root (default sibling repo ../LeetCode-Solutions).
   - Extracts exactly one Solution body per file (first occurrence), retaining leading comments/includes and helper structs/functions surrounding it.
   - Writes generated originals/variants/deviants to data/gen/{train,test}/ with metadata, plus JSONL datasets: triplets_train.jsonl, triplets_valid.jsonl, pairs_train.jsonl, pairs_test.jsonl, splits.json, metadata.jsonl.
   - Uses the same variant generator (tree-sitter + SemanticPreservingTransformation) and deviant mutator from [scripts/poj104/make_dataset.py](scripts/poj104/make_dataset.py).
   - Supports CLI knobs: dataset_root (abs/rel), output_root, train/test split counts or ratios, max files (default 20), kpos/kneg, seed, timeout, validate_candidates (defaults on with stub main), pilot/debug sampling.
2) README-style usage snippet documenting minimal commands to run the script on a small subset.

KEY DESIGN QUESTIONS
- How to guard against files without `class Solution`? -> skip with metadata reason.
- Compilation: Without LeetCode harness, most files will not compile. Default is best-effort stub compilation; can be disabled via flags.
- Multiple `Solution` occurrences: pick the first by source order; optionally log counts to metadata.
- Helpers defined after the class: keep entire file after extracting the first Solution to avoid dropping dependent code.

WORKPLAN
1) Survey a small sample of LeetCode files to confirm common patterns (multiple Solution classes, headers usage, helper structs present?).
2) Implement extraction utilities:
   - locate first `class Solution` (regex/tree-sitter), capture enclosing class definition; keep preamble and postamble text around that class to preserve helpers.
   - fallback: if parsing fails, store whole file but mark metadata reason.
3) Build CLI mirroring [scripts/poj104/make_dataset.py](scripts/poj104/make_dataset.py) with LeetCode defaults (validation on with stub main, dataset_root defaulted to sibling repo, max_files=20).
4) Integrate existing variant and deviant generation; default to stubbed compilation, allow skipping; emit metadata with compiler_ok/null when skipped.
5) Emit datasets and metadata; print stats and failure reasons; ensure deterministic splits via seed.
6) Smoke-test on a handful of files (train/test few each) and document commands in README snippet.
7) Refactor deviant mutation helpers (comment stripping, relational/equality/constant/arithmetic flips, mutation picker) out of [scripts/poj104/make_dataset.py](scripts/poj104/make_dataset.py) into a shared module (now [src/data_preprocessors/semantic_breaking_transformation.py](src/data_preprocessors/semantic_breaking_transformation.py)). Import that shared utility from both the POJ and new LeetCode dataset scripts to avoid duplication.

STATUS
- Script implemented and smoke-tested with defaults; output_root=data/leetcode, dataset_root=../LeetCode-Solutions. Latest run produced: triplets=14, pairs_train=28, pairs_test=8, metadata rows=97.

OUTPUT STRUCTURE (proposed)
- data/gen/{train,test}/{problem_like_id}/orig.cpp, *_v*.cpp, *_d*.cpp (problem id can be derived from filename stem).
- data/datasets/{triplets_train.jsonl, triplets_valid.jsonl, pairs_train.jsonl, pairs_test.jsonl, metadata.jsonl, splits.json}.

NOTES
- Keep tree_token_ids as 128 zeros, matching POJ.
- Preserve ASCII-only outputs; avoid introducing non-LeetCode headers unless needed for optional compilation.
