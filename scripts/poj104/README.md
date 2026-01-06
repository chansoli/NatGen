# POJ-104 NatGen data generation (pilot)

This folder provides a minimal data pipeline that builds triplets (pretrain) and pairs (fine-tune) from POJ-104 code using NatGen transformations and a differential-testing oracle.

## Step 0: select problems and solutions
The script samples problem IDs and 10 solutions per problem. Splits are stored in `data/datasets/splits.json`.

## Step 1 + 2: generate variants/deviants and write datasets
Pilot run (4 problems with input generators):

```
python scripts/poj104/make_dataset.py --pilot_only
```

Full 20-problem run (requires input generators for all 20 problems):

```
python scripts/poj104/make_dataset.py
```

Outputs:
- `data/gen/{train,test}/originals/`
- `data/gen/{train,test}/variants/`
- `data/gen/{train,test}/deviants/`
- `data/datasets/triplets_train.jsonl`
- `data/datasets/triplets_valid.jsonl`
- `data/datasets/pairs_train.jsonl`
- `data/datasets/pairs_test.jsonl`
- `data/datasets/metadata.jsonl`

Notes:
- Mutations skip code inside comments to avoid false-positive deviants (e.g., comment-only edits).
- A unit test suite covers mutation behaviors, including the comment-only case; run with `python -m pytest test/test_mutate_code.py`.
