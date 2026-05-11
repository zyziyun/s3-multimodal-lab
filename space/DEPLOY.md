# Deploying the demo to Hugging Face Spaces

The `space/` directory is a complete, deploy-ready Hugging Face Space.
It runs on **ZeroGPU** (free GPU quota) — ColPali needs a GPU to be usable.

## One-time setup

1. Create a Hugging Face account: <https://huggingface.co/join>
2. Generate a write token: <https://huggingface.co/settings/tokens>
3. Install the HF CLI and log in:
   ```bash
   pip install --upgrade huggingface_hub
   huggingface-cli login    # paste your token
   ```

## Create the Space

Pick a name (e.g. `s3-multimodal-lab-demo`) and create it:

```bash
huggingface-cli repo create s3-multimodal-lab-demo --type space --space_sdk gradio
```

In the Space's **Settings → Hardware**, switch to **"ZeroGPU"** (free).

## Push the code

From the repo root:

```bash
# Clone the new (empty) Space as a sibling repo
git clone https://huggingface.co/spaces/<your-hf-username>/s3-multimodal-lab-demo
cp space/{app.py,requirements.txt,packages.txt,README.md} s3-multimodal-lab-demo/
cd s3-multimodal-lab-demo
git add .
git commit -m "feat: initial deploy"
git push
```

Or, if you prefer, use the web UI: drag-and-drop the four files at
`https://huggingface.co/spaces/<you>/<space>/tree/main`.

First boot takes ~5 minutes (downloads the ~6 GB ColPali model into the
Space's persistent storage). Subsequent loads are fast.

## How users interact

1. Upload a PDF.
2. Paste their own OpenAI API key (this is the *only* sane pattern — you
   should never let strangers burn yours).
3. Click "Index PDF" — runs ColPali on ZeroGPU (~30 s for 20 pages).
4. Ask a question. Result: matched page image + grounded answer + confidence.

## Once it's live, link from the main repo README

Add this near the hero section of `README.md`:

```markdown
🚀 **[Try the live demo →](https://huggingface.co/spaces/<your-hf-username>/s3-multimodal-lab-demo)**
```
