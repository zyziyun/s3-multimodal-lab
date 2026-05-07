# Eval Suite

Per-phase ground truth + measurement runners. Each `run.py` produces a
report you can use to validate that a code change didn't regress quality.

## What's here

| Phase | Self-contained? | Metric | Pass criterion |
|-------|----------------|--------|---------------|
| 1 — Visual RAG (ColPali) | template | `recall@k` | `recall@3 ≥ 0.66` (default) |
| 3 — Code-switching STT | yes | avg CER + avg English recall | `en_recall ≥ 0.50` AND `cer ≤ 0.40` |
| 4 — Video retrieval | template | hit-within-tolerance | `hit_rate ≥ 0.66` |

(Phase 2 is multilingual Visual RAG — runs the Phase 1 harness against a
Chinese/Japanese PDF. Same code, different ground truth.)

## Run a single phase

```bash
# Phase 3 — runs out of the box (TTS-generated audio, just needs OPENAI_API_KEY)
python evals/phase3_codeswitch/run.py

# Phase 1 / 4 — copy the .example file, edit, run
cp evals/phase1_visual_rag/ground_truth.example.json evals/phase1_visual_rag/ground_truth.json
# edit `pdf` and `queries` ...
python evals/phase1_visual_rag/run.py
```

## Run everything

```bash
python evals/run_all.py
```

Skips templated phases that don't have a populated `ground_truth.json`.
Exits non-zero if any populated phase failed its threshold — wire into a
pre-merge check on quality-sensitive PRs.

## Reports

Reports land in `evals/reports/phase*.json` with row-level metrics, the
config used, and an aggregate. Diff reports between commits to see exactly
which sample regressed.

## Adding a new sample

For Phase 3:
1. Add an entry to `phase3_codeswitch/ground_truth.json` with a unique `id`,
   `text`, and `expected_english`.
2. Re-run — the runner will TTS the new sample on first run and cache it.

For Phase 1 / 4: edit your local `ground_truth.json`. Don't commit it
unless your PDF/video is also committed (which it shouldn't be — they're in
`.gitignore`).
