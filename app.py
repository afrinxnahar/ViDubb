"""Gradio UI for ViDubb. Pipeline logic lives in `tools.video_dubbing`."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from tools.video_dubbing import VideoDubbing

load_dotenv()

LANGUAGE_MAPPING: dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Hungarian": "hu",
}


def _uploaded_video_path(uploaded: Any) -> str | None:
    """Gradio 4/5 may pass a path string, dict, or FileData-like object."""
    if uploaded is None:
        return None
    if isinstance(uploaded, str) and uploaded.strip():
        return uploaded
    if isinstance(uploaded, dict):
        return (
            uploaded.get("path")
            or uploaded.get("name")
            or uploaded.get("video")
        )
    path = getattr(uploaded, "path", None)
    if path:
        return str(path)
    return None


def _resolve_input_video(uploaded_video: Any, youtube_url: str) -> str | None:
    if youtube_url and "youtube.com" in youtube_url:
        os.system(
            f'yt-dlp -f best -o "video_path.mp4" --recode-video mp4 {youtube_url}'
        )
        return "video_path.mp4"
    return _uploaded_video_path(uploaded_video)


def process_video(
    uploaded_video: Any,
    youtube_url: str,
    source_language: str,
    target_language: str,
    use_wav2lip: bool,
    whisper_model: str,
    bg_sound: bool,
) -> tuple[str | None, str]:
    try:
        if os.path.exists("video_path.mp4"):
            os.remove("video_path.mp4")
    except OSError:
        pass

    try:
        video_path = _resolve_input_video(uploaded_video, youtube_url or "")
        if not video_path:
            return None, "Error: Provide either a video file or a YouTube URL."

        hf = os.getenv("HF_TOKEN")
        # Match legacy app: Gradio always used MarianMT (7th arg was hard-coded "").
        VideoDubbing(
            video_path,
            LANGUAGE_MAPPING[source_language],
            LANGUAGE_MAPPING[target_language],
            lip_sync=use_wav2lip,
            voice_denoising=not bg_sound,
            whisper_model=whisper_model,
            context_translation="",
            huggingface_auth_token=hf,
        )

        if use_wav2lip:
            return "results/result_voice.mp4", "Done."
        if not bg_sound:
            return "results/denoised_video.mp4", "Done."
        return "results/output_video.mp4", "Done."
    except Exception as e:
        return None, f"Error: {e}"


with gr.Blocks(theme=gr.themes.Soft(), title="ViDubb") as demo:
    gr.Markdown("# ViDubb")
    gr.Markdown("AI-assisted video dubbing into another language.")

    with gr.Row():
        with gr.Column(scale=2):
            video_in = gr.Video(label="Upload video (optional)", height=500, width=500)
            youtube_url = gr.Textbox(
                label="YouTube URL (optional)",
                placeholder="https://www.youtube.com/...",
            )
            source_language = gr.Dropdown(
                choices=list(LANGUAGE_MAPPING.keys()),
                label="Source language",
                value="English",
            )
            target_language = gr.Dropdown(
                choices=list(LANGUAGE_MAPPING.keys()),
                label="Target language",
                value="French",
            )
            whisper_model = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                label="Whisper model",
                value="medium",
            )
            use_wav2lip = gr.Checkbox(
                label="Wav2Lip lip sync",
                value=False,
                info="Best for close-up faces; may fail on some clips.",
            )
            bg_sound = gr.Checkbox(
                label="Keep background sound",
                value=False,
                info="Keeps original bed; may add noise.",
            )
            submit = gr.Button("Process video", variant="primary")

        with gr.Column(scale=2):
            video_out = gr.Video(label="Output video", height=500, width=500)
            status = gr.Textbox(label="Status")

    submit.click(
        process_video,
        inputs=[
            video_in,
            youtube_url,
            source_language,
            target_language,
            use_wav2lip,
            whisper_model,
            bg_sound,
        ],
        outputs=[video_out, status],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
