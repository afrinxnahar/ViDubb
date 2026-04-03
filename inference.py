"""CLI entry: `python inference.py --video_url ...` or `--yt_url ...`."""

from __future__ import annotations

import argparse
import os

from ascii_magic import AsciiArt
from dotenv import load_dotenv

from tools.video_dubbing import VideoDubbing

load_dotenv()

_art = AsciiArt.from_image("Vidubb_without_bg.png")
_art.to_terminal()
print("Start processing...")


def main() -> None:
    parser = argparse.ArgumentParser(description="ViDubb CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--yt_url", type=str, default="", help="YouTube video URL")
    group.add_argument("--video_url", type=str, default="", help="Local or remote video path")

    parser.add_argument("--source_language", type=str, required=True, help="Source language code (e.g. en)")
    parser.add_argument("--target_language", type=str, required=True, help="Target language code (e.g. fr)")
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="medium",
        help="faster-whisper model size",
    )
    parser.add_argument("--LipSync", action="store_true", help="Enable Wav2Lip lip sync")
    parser.add_argument(
        "--Bg_sound",
        action="store_true",
        help="Keep background sound (omit for voice-only / denoise path)",
    )

    args = parser.parse_args()

    if os.path.exists("video_path.mp4"):
        try:
            os.remove("video_path.mp4")
        except OSError:
            pass

    video_path: str | None = None
    if args.yt_url:
        os.system(
            f'yt-dlp -f best -o "video_path.mp4" --recode-video mp4 {args.yt_url}'
        )
        video_path = "video_path.mp4"
    else:
        video_path = args.video_url

    if not video_path:
        raise SystemExit("No video path.")

    groq = os.getenv("GROQ_TOKEN") or os.getenv("Groq_TOKEN") or ""
    hf = os.getenv("HF_TOKEN")

    VideoDubbing(
        video_path,
        args.source_language,
        args.target_language,
        lip_sync=args.LipSync,
        voice_denoising=not args.Bg_sound,
        whisper_model=args.whisper_model,
        context_translation=groq,
        huggingface_auth_token=hf,
    )


if __name__ == "__main__":
    main()
