# Campaign Text + Image Generator (Render)

This deployment combines the assignment's text and image workflow. Campaign copy is generated with Groq. Poster backgrounds are generated remotely through Hugging Face Inference Providers, so the Render service does not need a local GPU or a locally loaded Stable Diffusion pipeline.

## Render settings
- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Instance: Free is suitable for a demo UI, but requests can be slow and free services may spin down.

## Required Render Environment Variables
- `GROQ_API_KEY`: your Groq API key
- `HF_TOKEN`: a Hugging Face user access token with permission to call Inference Providers
- Optional `GROQ_MODEL`, default `openai/gpt-oss-120b`
- Optional `HF_IMAGE_MODEL`, default `black-forest-labs/FLUX.1-schnell`

Never commit API keys into GitHub. The Hugging Face image inference call may require provider access/credits depending on your account and provider.
