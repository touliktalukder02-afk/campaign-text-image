import os
import textwrap
from pathlib import Path

import gradio as gr
from groq import Groq
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
OUTPUT_DIR = Path("/tmp/campaign_posters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise gr.Error("GROQ_API_KEY is missing. Add it in Render → Environment.")
    return Groq(api_key=key)


def hf_client():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise gr.Error("HF_TOKEN is missing. Add a Hugging Face token in Render → Environment.")
    return InferenceClient(api_key=token)


def generate_campaign(campaign_name, audience, objective, date, venue, tone):
    if not campaign_name or not campaign_name.strip():
        raise gr.Error("Enter a campaign or event name.")
    prompt = f"""Create a complete promotional campaign for this event.
Event/campaign: {campaign_name}
Target audience: {audience or 'College students'}
Objective: {objective or 'Create awareness'}
Date: {date or 'Not specified'}
Venue: {venue or 'Not specified'}
Tone: {tone or 'Modern technology'}

Provide:
1. Campaign overview and objective.
2. Poster headline and short description.
3. Instagram caption and five hashtags.
4. LinkedIn announcement.
5. WhatsApp promotional message.
6. Email subject and invitation.
7. Three promotional ideas.

Use clear, professional English. Do not invent event details, fees, speakers, or registration links. Mark missing details as not specified."""
    try:
        response = groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert college-event marketing strategist. Produce accurate, engaging campaign content."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_completion_tokens=1500,
        )
        return response.choices[0].message.content or "The model returned no text. Please try again."
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Groq text generation failed: {type(exc).__name__}: {str(exc)[:300]}")


def wrap_text(draw, text, font, max_width):
    words = str(text or "").split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def generate_poster(campaign_name, audience, objective, date, venue, tone, headline, description, call_to_action):
    if not campaign_name or not campaign_name.strip():
        raise gr.Error("Enter a campaign or event name first.")
    image_prompt = f"""Create a polished square Instagram promotional poster BACKGROUND for a college event called '{campaign_name}'.
Target audience: {audience or 'college students'}. Objective: {objective or 'create awareness'}.
Visual style: {tone or 'modern technology'}, professional, visually striking, sophisticated blue and purple palette, attractive lighting, abstract AI/technology shapes, clean composition, ample uncluttered lower half for text overlay.
Do not include any words, letters, logos, or watermark. Square 1:1 composition."""
    try:
        # Image generation is performed remotely by Hugging Face Inference Providers;
        # Render does not need to download or load a large diffusion model locally.
        image = hf_client().text_to_image(
            prompt=image_prompt,
            model=IMAGE_MODEL,
            num_inference_steps=4,
        )
        poster = image.convert("RGB").resize((1080, 1080))
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(
            "Image generation failed. Check HF_TOKEN permissions/Inference Providers access, "
            f"model availability, and provider credits. Details: {type(exc).__name__}: {str(exc)[:300]}"
        )

    draw = ImageDraw.Draw(poster, "RGBA")
    draw.rounded_rectangle((45, 610, 1035, 1035), radius=35, fill=(10, 18, 40, 232))
    title_font = get_font(54, bold=True)
    body_font = get_font(30)
    small_font = get_font(26, bold=True)

    title_lines = wrap_text(draw, (headline or campaign_name).strip(), title_font, 900)[:2]
    y = 640
    for line in title_lines:
        draw.text((85, y), line, font=title_font, fill=(255, 255, 255, 255))
        y += 68

    if description and description.strip():
        for line in wrap_text(draw, description.strip(), body_font, 900)[:1]:
            draw.text((88, y + 2), line, font=body_font, fill=(220, 235, 255, 255))
            y += 44

    for label, value in (("DATE", date), ("VENUE", venue)):
        if value and value.strip():
            line = f"{label}: {value.strip()}"
            for wrapped in wrap_text(draw, line, body_font, 900)[:1]:
                if y < 930:
                    draw.text((88, y + 3), wrapped, font=body_font, fill=(155, 220, 255, 255))
                    y += 42

    if call_to_action and call_to_action.strip():
        cta_lines = wrap_text(draw, call_to_action.strip(), small_font, 900)[:1]
        if cta_lines:
            draw.text((88, min(y + 4, 980)), cta_lines[0], font=small_font, fill=(255, 225, 130, 255))

    output_path = OUTPUT_DIR / "campaign_poster.png"
    poster.save(output_path, format="PNG")
    return poster, str(output_path)


with gr.Blocks(theme=gr.themes.Soft(), title="AI Campaign Text and Image Generator") as demo:
    gr.Markdown("""
    # 🎓 AI Campaign Text + Image Generator
    Generate campaign copy with **Groq** and create a poster background using **Hugging Face Inference Providers**.
    """)
    with gr.Row():
        with gr.Column():
            campaign_name = gr.Textbox(label="Campaign / Event Name", value="AI and Generative AI Workshop")
            audience = gr.Textbox(label="Target Audience", value="College students and beginners")
            objective = gr.Textbox(label="Campaign Objective", value="Encourage students to attend the AI workshop")
            event_date = gr.Textbox(label="Date", value="25 October 2026")
            venue = gr.Textbox(label="Venue", value="Computer Science Lab, VIT Vellore")
            tone = gr.Dropdown(
                label="Campaign Style",
                choices=["Modern technology", "Professional and corporate", "Creative and colorful", "Minimal and elegant", "Futuristic AI"],
                value="Modern technology",
            )
            generate_text_btn = gr.Button("Generate Campaign Content", variant="primary")
        with gr.Column():
            campaign_output = gr.Markdown(label="Generated Campaign Content")
    gr.Markdown("## Instagram Poster")
    headline = gr.Textbox(label="Poster Headline", value="UNLEASH THE FUTURE WITH AI")
    description = gr.Textbox(label="Short Description", value="Learn • Create • Innovate")
    cta = gr.Textbox(label="Call to Action", value="Join us! Register now!")
    generate_image_btn = gr.Button("Generate Instagram Poster", variant="primary")
    with gr.Row():
        poster_preview = gr.Image(label="Poster Preview", type="pil")
        poster_download = gr.File(label="Download Poster PNG")

    generate_text_btn.click(
        fn=generate_campaign,
        inputs=[campaign_name, audience, objective, event_date, venue, tone],
        outputs=campaign_output,
    )
    generate_image_btn.click(
        fn=generate_poster,
        inputs=[campaign_name, audience, objective, event_date, venue, tone, headline, description, cta],
        outputs=[poster_preview, poster_download],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
