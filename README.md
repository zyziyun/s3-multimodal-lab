# S3 Multimodal Lab — Visual RAG with ColPali

A small, hands-on lab that walks through modern multimodal AI by **building a Visual RAG system that can chat with chart-heavy PDFs**.

By the end, you will have built a pipeline that takes a PDF (financial report, research paper, slides), indexes every page **as an image** using ColPali, and answers questions like *"what was Q3 revenue?"* by retrieving the right page and feeding it to GPT-4o.

This is the architecture behind 2025-era document AI products. No OCR. No layout-destroying chunking. Just visual patches in, grounded answers out.

## Why this lab?

Traditional RAG over PDFs goes `parse → OCR → chunk → embed → search`. That pipeline destroys charts, tables, equations, figures — exactly the parts of a technical document that carry the most information.

**ColPali** (Faysse et al., 2024) treats each page as an image, produces ~1024 patch embeddings per page, and uses **late interaction (MaxSim)** to score query tokens against page patches. Every query token finds its best-matching patch. Visual structure is preserved.

## Notebooks

Run them in order. Each one stands alone but builds on the previous concept.

| # | Notebook | What you'll learn | Maps to S3 § |
|---|----------|-------------------|--------------|
| 01 | [How VLMs See Images](notebooks/01_how_vlms_see_images.ipynb) | Image tokens, GPT-4o cost math, why resolution matters | §1.3, §2 |
| 02 | [CLIP Text→Image Search](notebooks/02_clip_text_to_image_search.ipynb) | Multimodal embeddings, contrastive learning, the modality gap | §4.1, §4.2 |
| 03 | [ColPali Visual RAG](notebooks/03_colpali_visual_rag.ipynb) | Late interaction, MaxSim, end-to-end visual document RAG | §4.3, §5.3 |
| 04 | [Grounded Generation](notebooks/04_grounded_generation.ipynb) | Strict schema, visual evidence, hallucination mitigation | §7 |
| 05 | [Multilingual Visual RAG](notebooks/05_multilingual_visual_rag.ipynb) | Why ColPali beats OCR on Chinese/Japanese PDFs | §4.3 |

## Setup

### 1. Clone & install

```bash
git clone <this-repo>
cd s3-multimodal-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. System dependencies

`pdf2image` needs `poppler`. Notebook 05 also needs `tesseract` with Chinese & Japanese language packs for the OCR baseline.

```bash
# macOS
brew install poppler tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-jpn
```

### 3. API keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 4. Sample PDFs

Drop any PDF you want to chat with into `data/sample_pdfs/`. A chart-heavy one (financial report, paper with figures, technical slide deck) shows ColPali's strengths most clearly.

## Hardware notes

- **Notebooks 01, 02, 04** run fine on CPU.
- **Notebook 03 (ColPali)** is memory-heavy. Recommended:
  - macOS: Apple Silicon with ≥16 GB RAM (uses MPS backend)
  - Linux: NVIDIA GPU with ≥10 GB VRAM
  - Or run on Google Colab with T4/A100

If ColPali is too heavy for your setup, the notebook includes a simpler CLIP-based fallback so you still get the end-to-end story.

## What you'll have built at the end

A working Visual RAG service that:
- Indexes PDFs as images (no OCR)
- Retrieves the right page for any question — including questions about specific chart bars or table cells
- Generates answers grounded in the retrieved page, with a strict JSON schema and visual-evidence citations to suppress hallucination
- Logs token usage so you can reason about cost in production

Résumé-quality work.

## References

- [ColPali paper (Faysse et al., 2024)](https://arxiv.org/abs/2407.01449)
- [Hugging Face — VLMs in 2025 overview](https://huggingface.co/blog/vlms-2025)
- [OpenAI Images and Vision guide](https://platform.openai.com/docs/guides/images-vision)
- [Vidore ColPali model card](https://huggingface.co/vidore/colpali-v1.3-hf)
