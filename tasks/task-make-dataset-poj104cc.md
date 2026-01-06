You are Copilot Agent working in a C++ + POJ-104 environment. Implement the absolute-minimum experiment to measure the effect of CONCORD-style contrastive pretraining on distinguishing NatGen variants vs deviants.

GOAL
- Compare two conditions on the SAME downstream task:
  (A) Base model = microsoft/codebert-base -> fine-tune classifier
  (B) Base model -> CONCORD-style triplet pretrain on (orig, variant, deviant) -> fine-tune same classifier
- Evaluate on held-out POJ problems to see variant vs deviant gap reduction (especially deviant recall).

CONSTRAINTS
- Language: C++
- Dataset: POJ-104 accepted solutions
- Variants: generated with NatGen (existing in our checkout)
- Deviants: extend NatGen with minimal semantics-breaking mutations + differential testing oracle
- Use max_seq_length=128, keep tree_token_ids as all zeros (length=128)
- Keep experiment SMALL (fast): 20 POJ problems, 10 solutions/problem = 200 originals
- Split by problem IDs: 16 train problems, 4 test problems (no overlap)
- Each original: Kpos=2 variants, Kneg=2 deviants; keep only those validated by oracle

DELIVERABLES (FILES + COMMANDS)
1) Data generation pipeline that outputs three datasets:
   Dataset A (triplets, for contrastive pretrain):
     JSONL lines: {"orig_code":..., "positive_code":..., "negative_code":..., "tree_token_ids":"0 0 ... (128 zeros)"}
   Dataset B (train pairs, for fine-tune):
     JSONL lines: {"orig_code":..., "cand_code":..., "label":1 or 0}
   Dataset C (test pairs, same schema as B, held-out problems)

2) Minimal deviant generator:
   - Use tree-sitter C++ parser OR simple token-based mutation (absolute minimum acceptable)
   - Mutations (apply exactly one per mutant attempt):
     * relational flip: <-><=, >->>=
     * equality flip: ==<->!=
     * constant perturbation: integer literal +/- 1
     * arithmetic flip: +<->- (avoid loop index if possible; otherwise allow)
   - Compile candidate (g++ -O2 -std=c++17); discard if compile fails
   - Differential testing oracle vs original:
     * For each problem, generate M=30 random tests per original using a per-problem input generator
     * Run both binaries; if any test output differs -> label as deviant (accept)
     * If all outputs match -> treat as variant-equivalent and reject from deviant pool

3) Minimal variant generator hookup:
   - Use NatGen to generate semantics-preserving variants; integrate with the same oracle:
     * accept as variant only if ALL M tests match original outputs
     * compile must succeed

4) Training scripts:
   - Pretrain (CONCORD-style contrastive) on Dataset A:
     * Start from microsoft/codebert-base
     * 1 epoch, batch 8-16, fp16 if available
     * Loss: triplet loss or InfoNCE where anchor=orig, pos=variant, neg=deviant
     * Output: pretrained checkpoint dir
   - Fine-tune classifier on Dataset B for BOTH conditions:
     * Condition A: start from microsoft/codebert-base
     * Condition B: start from pretrained checkpoint
     * Encoder pooling: CLS or mean pooling
     * Classifier input: concat(e_orig, e_cand, |e_orig-e_cand|)
     * 3 epochs, lr ~2e-5, batch 16, max_len 128
   - Evaluate on Dataset C:
     * Report accuracy, deviant recall, deviant F1

5) A single top-level README-style set of commands:
   - Step 0: select subset of POJ problems and solutions
   - Step 1: generate variants/deviants + label using oracle
   - Step 2: write Dataset A/B/C
   - Step 3: run pretrain (for condition B)
   - Step 4: run fine-tune for condition A and B
   - Step 5: print metrics side-by-side and the delta

IMPLEMENTATION DETAILS
- Directory layout:
  data/raw/poj104/...
  data/gen/{train,test}/originals/
  data/gen/{train,test}/variants/
  data/gen/{train,test}/deviants/
  data/datasets/
    triplets_train.jsonl
    triplets_valid.jsonl (optional; can reuse train)
    pairs_train.jsonl
    pairs_test.jsonl
- Use deterministic seeds everywhere (Python, numpy, torch).
- Ensure no leakage: split by problem_id BEFORE generating triplets/pairs; never mix train/test originals.
- Store metadata per sample:
  {"problem_id":..., "orig_id":..., "variant_id":..., "deviant_id":..., "compiler_ok":..., "oracle_tests":..., "diff_found":...}
  (can be separate log file) so we can debug.

WHAT TO DO FIRST
- Inspect current repo for NatGen entrypoints we already have; locate how variants are generated.
- Implement per-problem input generators for the 20 selected POJ problems. Start with 4 problems to validate end-to-end, then scale to 20.
- Build the dataset writers (triplets + pairs) once the generation works.

SUCCESS CRITERIA
- Base (A) should show high performance on variants but low deviant recall.
- CONCORD-style (B) should materially improve deviant recall/F1 while keeping variant accuracy similar.
- Output a small table:
  Model | Acc | Deviant Recall | Deviant F1
  CodeBERT | ... | ... | ...
  CodeBERT+CONCORD-pretrain | ... | ... | ...

Do not overbuild. This is minimum viable; prioritize correctness and separation of Dataset A vs Dataset B/C.
