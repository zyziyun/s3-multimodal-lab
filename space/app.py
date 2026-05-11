"""S3 Multimodal Lab — Visual RAG demo on Hugging Face Spaces.

Hosted version of nb03 + nb04: upload a PDF, ask a question, get a
grounded answer with the matched page image and a verbatim
visual-evidence quote.

Runs on a ZeroGPU-equipped Space (uses @spaces.GPU on the heavy paths).
User brings their own OpenAI API key — nothing is logged or persisted.
"""

from __future__ import annotations
import base64
import io
import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import ColPaliForRetrieval, ColPaliProcessor
from openai import OpenAI


MODEL_NAME = "vidore/colpali-v1.3-hf"
MAX_PAGES = 25            # cap to keep ZeroGPU runtime under the per-call quota
CONFIDENCE_THRESHOLD = 0.4


# Load model + processor at startup (on CPU). The @spaces.GPU functions
# below move the model to CUDA inside the GPU quota window.
print("loading ColPali...")
model = ColPaliForRetrieval.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
print("ready.")


def _maxsim(q: torch.Tensor, p: torch.Tensor) -> float:
    return float((q.float() @ p.float().T).max(dim=1).values.sum())


@spaces.GPU(duration=180)
def index_pdf(pdf_path):
    """Render + embed pages. Returns (state, status_message)."""
    if pdf_path is None:
        return gr.skip(), "Upload a PDF first."

    try:
        pages = convert_from_path(pdf_path, dpi=150, last_page=MAX_PAGES)
    except Exception as e:
        return gr.skip(), f"Failed to read PDF: {e}"

    model.to("cuda")
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(pages), 4):
            batch = pages[i : i + 4]
            inputs = processor(images=batch, return_tensors="pt").to("cuda")
            for emb in model(**inputs).embeddings:
                embeddings.append(emb.cpu())
    model.to("cpu")

    state = {"pages": pages, "embeddings": embeddings}
    return state, f"✅ Indexed {len(pages)} page(s). Ask a question below."


@spaces.GPU(duration=60)
def _embed_query_on_gpu(query: str) -> torch.Tensor:
    model.to("cuda")
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt").to("cuda")
        emb = model(**inputs).embeddings[0].cpu()
    model.to("cpu")
    return emb


def search_and_answer(state, query, api_key):
    """Retrieve top page with ColPali, then send to GPT-4o-mini for grounded answer."""
    if not state:
        return None, "Upload + index a PDF first."
    if not query or not query.strip():
        return None, "Type a question."
    if not api_key or not api_key.strip():
        return None, "Paste your OpenAI API key (stored only in-memory for this turn)."

    pages = state["pages"]
    page_embs = state["embeddings"]

    q_emb = _embed_query_on_gpu(query)
    scored = sorted(
        ((_maxsim(q_emb, pe), idx) for idx, pe in enumerate(page_embs)),
        reverse=True,
    )
    top_idx = scored[0][1]
    top_page = pages[top_idx]

    # Grounded VLM call
    buf = io.BytesIO()
    top_page.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    system = (
        "You answer questions about the provided PDF page image. "
        "Respond as JSON with keys: answer (string or null), "
        "evidence (verbatim quote / specific element observed), "
        "confidence (0.0-1.0). "
        "If the answer is not visible on the page, set answer to null and "
        "confidence below 0.3."
    )
    user_text = f"Question: {query}"

    try:
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "page_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["answer", "evidence", "confidence"],
                        "properties": {
                            "answer":     {"type": ["string", "null"]},
                            "evidence":   {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                    },
                },
            },
        )
    except Exception as e:
        return top_page, f"OpenAI call failed: {e}"

    import json
    parsed = json.loads(r.choices[0].message.content)
    conf = parsed["confidence"]
    answer = parsed["answer"]
    evidence = parsed["evidence"]

    if conf < CONFIDENCE_THRESHOLD or answer is None:
        body = (
            f"### 🛑 Refused (confidence {conf:.2f} below threshold)\n\n"
            f"**Reason:** {evidence}\n\n"
            f"_Matched page {top_idx + 1} but the answer isn't grounded enough to return safely._"
        )
    else:
        body = (
            f"### 📄 Page {top_idx + 1}  ·  confidence {conf:.2f}\n\n"
            f"**Answer:** {answer}\n\n"
            f"**Visual evidence:** _{evidence}_"
        )
    return top_page, body


with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="S3 Multimodal Lab Demo") as demo:
    gr.Markdown(
        """
        # 📚 S3 Multimodal Lab — Visual RAG with ColPali
        Upload a PDF (financial report, slide deck, paper with figures), ask a question.
        ColPali retrieves the right page (no OCR — works on charts and tables natively).
        GPT-4o-mini answers, citing what it actually saw.

        🔗 [Full 12-notebook walkthrough on GitHub](https://github.com/zyziyun/s3-multimodal-lab)
        """
    )

    state = gr.State()

    with gr.Row():
        with gr.Column(scale=2):
            pdf_in = gr.File(
                label="Upload PDF (first 25 pages indexed)",
                file_types=[".pdf"], type="filepath",
            )
            index_btn = gr.Button("Index PDF", variant="primary")
            status = gr.Markdown()
        with gr.Column(scale=3):
            api_key = gr.Textbox(
                label="OpenAI API key",
                placeholder="sk-...",
                type="password",
                info="Used only for this session. Never stored.",
            )
            query = gr.Textbox(
                label="Your question",
                placeholder="What was Q3 revenue? / 营业收入趋势如何？",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        page_out = gr.Image(label="Matched page", height=500)
        answer_out = gr.Markdown(label="Grounded answer")

    gr.Examples(
        examples=[
            ["What is the title of the document?"],
            ["Summarize the main findings."],
            ["What is the revenue growth rate?"],
            ["这份文件的主要主题是什么？"],
        ],
        inputs=[query],
    )

    index_btn.click(index_pdf, inputs=[pdf_in], outputs=[state, status])
    ask_btn.click(search_and_answer, inputs=[state, query, api_key], outputs=[page_out, answer_out])

    gr.Markdown(
        """
        ---
        **What's not in this demo (but is in the lab):** voice in/out, code-switching
        speech, video search, native long-context comparison, full agent capstone with
        confidence gating, eval harness. See the
        [GitHub repo](https://github.com/zyziyun/s3-multimodal-lab).
        """
    )


if __name__ == "__main__":
    demo.launch()
