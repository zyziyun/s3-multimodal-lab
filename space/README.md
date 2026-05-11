---
title: S3 Multimodal Lab — Visual RAG Demo
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Chat with chart-heavy PDFs using ColPali + GPT-4o (no OCR)
---

# S3 Multimodal Lab — Visual RAG Demo

Drop a PDF, ask a question, get an answer grounded in the right page.
Uses **ColPali** (late interaction over patch embeddings) for retrieval
and **GPT-4o** for grounded generation. **No OCR** in the pipeline.

This is a hosted demo of [the full lab](https://github.com/zyziyun/s3-multimodal-lab).

## How it works

```
PDF -> render pages -> ColPali (~1024 patch vectors/page)
                          ↓
question -> ColPali query embed -> MaxSim late interaction -> top page
                                          ↓
                                  GPT-4o-mini w/ page image
                                          ↓
                                  grounded answer + evidence
```

You bring your own OpenAI API key (paste in the UI). Inputs are not stored.

## Why this exists

Traditional `parse → OCR → chunk → embed → search` destroys exactly the
parts of a technical document that carry the most information: charts,
tables, equations, figures. ColPali keeps the page as an image, embeds
every patch, and uses late interaction so a query token like *"Q3 revenue"*
can match the *specific bar* on the chart — not the whole page averaged
together.

See the [12-notebook walkthrough](https://github.com/zyziyun/s3-multimodal-lab)
for the full story: voice agents, code-switching, video search, and
production safety patterns.
