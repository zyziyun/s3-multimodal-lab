"""Phase 1 — ColPali retrieval quality eval.

Reports recall@k for a configured query set against a configured PDF.
Requires user-supplied ground truth (copy ground_truth.example.json,
edit, run).

Usage:
    python evals/phase1_visual_rag/run.py
    python evals/phase1_visual_rag/run.py --ground-truth path/to/gt.json
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.common import EvalReport, load_ground_truth, fail


HERE = Path(__file__).parent
DEFAULT_GT = HERE / "ground_truth.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--report-path", type=Path,
                        default=REPO_ROOT / "evals" / "reports" / "phase1.json")
    args = parser.parse_args()

    if not args.ground_truth.exists():
        fail(
            f"{args.ground_truth} not found.\n"
            f"Copy {HERE / 'ground_truth.example.json'} to ground_truth.json,\n"
            f"point `pdf` at your file, edit `queries` with expected page indices, then re-run."
        )

    gt = load_ground_truth(args.ground_truth)
    pdf_path = REPO_ROOT / gt["pdf"]
    queries = gt["queries"]
    ks = gt.get("metrics", {}).get("recall_at_k", [1, 3, 5])

    if not pdf_path.exists():
        fail(f"PDF not found: {pdf_path}")

    print(f"Phase 1 eval — PDF={pdf_path.name}, {len(queries)} queries, recall@{ks}")

    # Heavy imports deferred so the structure test can validate without them
    import torch
    from pdf2image import convert_from_path
    from transformers import ColPaliForRetrieval, ColPaliProcessor

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading ColPali on {device}...")
    model = ColPaliForRetrieval.from_pretrained(
        "vidore/colpali-v1.3-hf", torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    proc = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")

    pages = convert_from_path(str(pdf_path), dpi=150, last_page=gt.get("max_pages", 30))
    print(f"  rendered {len(pages)} pages")

    @torch.no_grad()
    def embed_pages(pil_pages):
        out = []
        for i in range(0, len(pil_pages), 4):
            b = pil_pages[i : i + 4]
            inp = proc(images=b, return_tensors="pt").to(device)
            for emb in model(**inp).embeddings:
                out.append(emb.cpu())
        return out

    @torch.no_grad()
    def embed_query(q):
        return model(**proc(text=[q], return_tensors="pt").to(device)).embeddings[0].cpu()

    def maxsim(q, p):
        return float((q.float() @ p.float().T).max(dim=1).values.sum())

    def search(q, page_embs, k):
        qv = embed_query(q)
        scored = sorted(((maxsim(qv, pe), idx) for idx, pe in enumerate(page_embs)), reverse=True)
        return [idx for _, idx in scored[:k]]

    page_embs = embed_pages(pages)

    rep = EvalReport(phase="phase1_visual_rag", config={"pdf": str(pdf_path.name), "ks": ks})
    for q in queries:
        topk = search(q["q"], page_embs, k=max(ks))
        expected = set(q["expected_pages"])
        # Ground truth is human-readable 1-indexed; convert.
        expected_idx = {p - 1 for p in expected}
        metrics = {
            f"recall@{k}": float(any(r in expected_idx for r in topk[:k]))
            for k in ks
        }
        rep.add(
            sample_id=q["q"][:30],
            metrics=metrics,
            note=f"top1={topk[0] + 1}  expected={sorted(expected)}",
        )
    rep.aggregate_mean()
    rep.print()

    th_3 = gt.get("thresholds", {}).get("min_recall_at_3", 0.0)
    passed = rep.aggregate.get("recall@3", 0.0) >= th_3
    print(f"\nrecall@3 = {rep.aggregate.get('recall@3', 0.0):.3f}  "
          f"threshold = {th_3:.3f}  →  {'PASS' if passed else 'fail'}")

    rep.save(args.report_path)
    print(f"report saved to {args.report_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
