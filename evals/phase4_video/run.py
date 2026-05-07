"""Phase 4 — video RRF-fused retrieval timestamp eval.

For each (query, expected_timestamp_s, tolerance_s), search the dual index
and count a hit if the top fused timestamp is within tolerance.

Usage:
    python evals/phase4_video/run.py
    python evals/phase4_video/run.py --ground-truth path/to/gt.json
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.common import EvalReport, load_ground_truth, fail
from s3lab.retrieval import rrf_fuse


HERE = Path(__file__).parent
DEFAULT_GT = HERE / "ground_truth.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--report-path", type=Path,
                        default=REPO_ROOT / "evals" / "reports" / "phase4.json")
    args = parser.parse_args()

    if not args.ground_truth.exists():
        fail(
            f"{args.ground_truth} not found.\n"
            f"Copy {HERE / 'ground_truth.example.json'} to ground_truth.json,\n"
            f"point `video` at your file, edit `queries` + timestamps, then re-run."
        )
    if not shutil.which("ffmpeg"):
        fail("ffmpeg not in PATH (brew install ffmpeg)")

    gt = load_ground_truth(args.ground_truth)
    video_path = REPO_ROOT / gt["video"]
    queries = gt["queries"]
    if not video_path.exists():
        fail(f"video not found: {video_path}")

    print(f"Phase 4 eval — video={video_path.name}, {len(queries)} queries")

    # Heavy imports
    import numpy as np
    import torch, torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from faster_whisper import WhisperModel
    from sentence_transformers import SentenceTransformer

    work = video_path.parent / "_eval_work"
    work.mkdir(exist_ok=True)
    frames_dir = work / "frames"
    audio_path = work / "audio.wav"
    frames_dir.mkdir(exist_ok=True)
    if not list(frames_dir.glob("*.jpg")):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                        "-vf", "fps=1", "-q:v", "4",
                        str(frames_dir / "frame_%05d.jpg")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                        "-ac", "1", "-ar", "16000", str(audio_path)], check=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    frame_seconds = [int(p.stem.split("_")[1]) for p in frame_paths]

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def img_vecs(paths):
        out = []
        for i in range(0, len(paths), 16):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i+16]]
            inp = cproc(images=imgs, return_tensors="pt").to(device)
            out.append(F.normalize(clip.get_image_features(**inp), dim=-1).cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def txt_vec(t):
        inp = cproc(text=[t], return_tensors="pt", padding=True).to(device)
        return F.normalize(clip.get_text_features(**inp), dim=-1).cpu()[0]

    print("indexing frames...")
    frame_vecs = img_vecs(frame_paths)

    print("transcribing audio...")
    asr = WhisperModel("large-v3", device="cpu", compute_type="int8")
    seg, _ = asr.transcribe(str(audio_path), vad_filter=True)
    audio_segments = list(seg)
    audio_starts = [s.start for s in audio_segments]
    audio_texts = [s.text.strip() for s in audio_segments]

    st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    audio_vecs = st.encode(audio_texts, normalize_embeddings=True, show_progress_bar=False)

    def search_frame_ts(q, k=10):
        sims = (frame_vecs @ txt_vec(q)).numpy()
        idx = np.argsort(-sims)[:k]
        return [frame_seconds[i] for i in idx]

    def search_audio_ts(q, k=10):
        qv = st.encode([q], normalize_embeddings=True)[0]
        idx = np.argsort(-(audio_vecs @ qv))[:k]
        return [audio_starts[i] for i in idx]

    rep = EvalReport(phase="phase4_video", config={"video": video_path.name})
    for q in queries:
        f_ts = search_frame_ts(q["q"], k=10)
        a_ts = search_audio_ts(q["q"], k=10)
        fused = rrf_fuse([f_ts, a_ts], bucket=5)
        if not fused:
            top = -1.0
        else:
            top = float(fused[0][0])
        hit = abs(top - q["timestamp_s"]) <= q["tolerance_s"]
        rep.add(
            sample_id=q["q"][:30],
            metrics={
                "top_ts_s": top,
                "expected_ts_s": q["timestamp_s"],
                "abs_err_s": abs(top - q["timestamp_s"]),
                "hit": float(hit),
            },
        )
    rep.aggregate_mean()
    rep.print()

    th = gt.get("thresholds", {}).get("min_hit_rate", 0.0)
    rate = rep.aggregate.get("hit", 0.0)
    print(f"\nhit-rate = {rate:.3f}  threshold = {th:.3f}  →  {'PASS' if rate >= th else 'fail'}")

    rep.save(args.report_path)
    print(f"report saved to {args.report_path}")
    sys.exit(0 if rate >= th else 1)


if __name__ == "__main__":
    main()
