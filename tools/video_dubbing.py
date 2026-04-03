"""Video dubbing pipeline (speaker diarization, ASR, translation, TTS, mux).

Uses pyannote.audio 4.x community pipeline on Hugging Face (`token=`, not `use_auth_token`).
See: https://pypi.org/project/pyannote-audio/ and HF hub model cards.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from typing import Any, Iterator, Tuple

import cv2
import nltk
import torch
from audio_separator.separator import Separator
from faster_whisper import WhisperModel
from groq import Groq
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from pyannote.audio import Pipeline
from speechbrain.inference.interfaces import foreign_class
from TTS.api import TTS
from transformers import MarianMTModel, MarianTokenizer

from tools.utils import (
    detect_and_crop_faces,
    extract_and_save_most_common_face,
    extract_frames,
    get_overlap,
    get_speaker,
)

warnings.filterwarnings("ignore")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _reset_workdirs() -> None:
    for path in ("audio", "results"):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


def _iter_diarization_segments(diarization_output: Any) -> Iterator[Tuple[Any, str]]:
    """Support pyannote.audio 4.x `DiarizeOutput.speaker_diarization` and Annotation-style APIs."""
    ann = getattr(diarization_output, "speaker_diarization", diarization_output)
    if hasattr(ann, "itertracks"):
        for segment, _track, speaker in ann.itertracks(yield_label=True):
            yield segment, speaker
        return
    for item in ann:
        if len(item) == 2:
            yield item[0], item[1]


def _run_ffmpeg_mux(video_path: str, audio_path: str, out_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            out_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_wav2lip(face_video: str, audio_wav: str) -> None:
    root = os.path.abspath("Wav2Lip")
    subprocess.run(
        [
            sys.executable,
            "inference.py",
            "--checkpoint_path",
            "wav2lip_gan.pth",
            "--face",
            os.path.abspath(face_video),
            "--audio",
            os.path.abspath(audio_wav),
            "--face_det_batch_size",
            "1",
            "--wav2lip_batch_size",
            "1",
        ],
        cwd=root,
        check=False,
    )


class VideoDubbing:
    """End-to-end dubbing: diarize → transcribe → translate → synthesize → mux (optional Wav2Lip)."""

    def __init__(
        self,
        video_path: str,
        source_language: str,
        target_language: str,
        lip_sync: bool = True,
        voice_denoising: bool = True,
        whisper_model: str = "medium",
        context_translation: str = "",
        huggingface_auth_token: str | None = None,
        marian_on_cpu: bool = False,
    ) -> None:
        self.video_path = video_path
        self.source_language = source_language
        self.target_language = target_language
        self.lip_sync = lip_sync
        self.voice_denoising = voice_denoising
        self.whisper_model = whisper_model
        self.context_translation = context_translation
        self.huggingface_auth_token = huggingface_auth_token
        self.marian_on_cpu = marian_on_cpu

        _reset_workdirs()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        hf_token: str | bool | None = huggingface_auth_token or True
        if isinstance(huggingface_auth_token, str) and not huggingface_auth_token.strip():
            hf_token = True

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        ).to(device)

        audio_seg = AudioSegment.from_file(self.video_path, format="mp4")
        audio_file = "audio/test0.wav"
        audio_seg.export(audio_file, format="wav")

        diar_out = pipeline(audio_file)
        speakers_rolls: dict[tuple[float, float], str] = {}
        for speech_turn, speaker in _iter_diarization_segments(diar_out):
            if abs(speech_turn.end - speech_turn.start) > 1.5:
                print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
                speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker

        if self.lip_sync:
            video = cv2.VideoCapture(self.video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()

            frame_per_speaker: list[str | None] = []
            for i in range(total_frames):
                t = i / round(fps) if fps else 0.0
                frame_per_speaker.append(get_speaker(t, speakers_rolls))

            shutil.rmtree("speakers_image", ignore_errors=True)
            os.makedirs("speakers_image", exist_ok=True)
            extract_frames(self.video_path, "speakers_image", speakers_rolls)

            haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(haar)

            for speaker_folder in os.listdir("speakers_image"):
                speaker_folder_path = os.path.join("speakers_image", speaker_folder)
                if not os.path.isdir(speaker_folder_path):
                    continue
                for image_name in os.listdir(speaker_folder_path):
                    image_path = os.path.join(speaker_folder_path, image_name)
                    if not detect_and_crop_faces(image_path, face_cascade):
                        os.remove(image_path)
                        print(f"Deleted {image_path} due to no face detected.")
                    else:
                        print(f"Face detected and cropped: {image_path}")

            for speaker_folder in os.listdir("speakers_image"):
                speaker_folder_path = os.path.join("speakers_image", speaker_folder)
                if os.path.isdir(speaker_folder_path):
                    print(f"Processing images in folder: {speaker_folder}")
                    extract_and_save_most_common_face(speaker_folder_path)

            for root, _dirs, files in os.walk("speakers_image"):
                for file in files:
                    if file != "max_image.jpg":
                        os.remove(os.path.join(root, file))

            with open("frame_per_speaker.json", "w", encoding="utf-8") as f:
                json.dump(frame_per_speaker, f)

            if os.path.exists("Wav2Lip/frame_per_speaker.json"):
                os.remove("Wav2Lip/frame_per_speaker.json")
            shutil.copyfile("frame_per_speaker.json", "Wav2Lip/frame_per_speaker.json")

            if os.path.exists("Wav2Lip/speakers_image"):
                shutil.rmtree("Wav2Lip/speakers_image")
            shutil.copytree("speakers_image", "Wav2Lip/speakers_image")

        shutil.rmtree("speakers_audio", ignore_errors=True)
        os.makedirs("speakers_audio", exist_ok=True)

        speakers = set(speakers_rolls.values())
        src_audio = AudioSegment.from_file(audio_file, format="mp4")
        for speaker in speakers:
            speaker_audio = AudioSegment.empty()
            for key, value in speakers_rolls.items():
                if speaker == value:
                    start = int(key[0]) * 1000
                    end = int(key[1]) * 1000
                    speaker_audio += src_audio[start:end]
            speaker_audio.export(f"speakers_audio/{speaker}.wav", format="wav")

        most_occurred_speaker = max(
            list(speakers_rolls.values()),
            key=list(speakers_rolls.values()).count,
        )

        asr = WhisperModel(
            self.whisper_model,
            device="cuda" if use_cuda else "cpu",
            compute_type="float16" if use_cuda else "default",
        )
        segments, _info = asr.transcribe(self.video_path, word_timestamps=True)
        segments = list(segments)

        time_stamped: list[list[Any]] = []
        full_text: list[str] = []
        for segment in segments:
            if segment.words is None:
                continue
            for word in segment.words:
                time_stamped.append([word.word, word.start, word.end])
                full_text.append(word.word)
        full_text_str = "".join(full_text)

        tokenized_sentences = sent_tokenize(full_text_str)
        sentences = list(tokenized_sentences)

        time_stamped_sentences: dict[str, list[float]] = {}
        letter = 0
        for i in range(len(sentences)):
            tmp: list[str] = []
            starts: list[float] = []
            for _j in range(len(sentences[i])):
                letter += 1
                tmp.append(sentences[i][_j])
                f = 0
                for k in range(len(time_stamped)):
                    for _m in range(len(time_stamped[k][0])):
                        f += 1
                        if f == letter:
                            starts.append(time_stamped[k][1])
                            starts.append(time_stamped[k][2])
            letter += 1
            key = "".join(tmp)
            time_stamped_sentences[key] = [min(starts), max(starts)]

        record: list[list[Any]] = []
        for sentence in time_stamped_sentences:
            record.append([sentence, time_stamped_sentences[sentence][0], time_stamped_sentences[sentence][1]])

        new_record = record

        classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": str(device)},
        )

        emotion_dict = {
            "neu": "Neutral",
            "ang": "Angry",
            "hap": "Happy",
            "sad": "Sad",
            "None": None,
        }

        if not self.context_translation:

            def translate(sentence: str) -> str:
                if self.source_language == "tr":
                    model_name = f"Helsinki-NLP/opus-mt-trk-{self.target_language}"
                elif self.target_language == "tr":
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-trk"
                elif self.source_language == "zh-cn":
                    model_name = f"Helsinki-NLP/opus-mt-zh-{self.target_language}"
                elif self.target_language == "zh-cn":
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-zh"
                else:
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{self.target_language}"

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                mt_device = torch.device("cpu") if self.marian_on_cpu else device
                mt_model = MarianMTModel.from_pretrained(model_name).to(mt_device)
                inputs = tokenizer([sentence], return_tensors="pt", padding=True).to(mt_device)
                out = mt_model.generate(**inputs)
                return tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            client = Groq(api_key=self.context_translation)

            def translate(
                sentence: str,
                before_context: str,
                after_context: str,
                target_language: str,
            ) -> str:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Role: You are a professional translator who translates concisely in short "
                                f"sentences while preserving meaning.\nTranslate into {target_language}.\n"
                                f"Context before: {before_context}\n"
                                f"Sentence: {sentence}\n"
                                f"Context after: {after_context}\n"
                                "Output format: [[sentence translation: <your translation>]]"
                            ),
                        }
                    ],
                    model="llama3-70b-8192",
                )
                pattern = r"\[\[sentence translation: (.*?)\]\]"
                match = re.search(pattern, chat_completion.choices[0].message.content or "")
                if match:
                    return match.group(1)
                return sentence

        records: list[list[Any]] = []
        seg_audio = AudioSegment.from_file(audio_file, format="mp4")
        for i in range(len(new_record)):
            final_sentence = new_record[i][0]
            if not self.context_translation:
                translated = translate(sentence=final_sentence)
            else:
                before = new_record[i - 1][0] if i - 1 in range(len(new_record)) else ""
                after = new_record[i + 1][0] if i + 1 in range(len(new_record)) else ""
                translated = translate(
                    sentence=final_sentence,
                    before_context=before,
                    after_context=after,
                    target_language=self.target_language,
                )
            speaker = most_occurred_speaker
            max_overlap = 0.0
            for key, value in speakers_rolls.items():
                sp_start = int(key[0])
                sp_end = int(key[1])
                overlap = get_overlap(
                    (new_record[i][1], new_record[i][2]),
                    (sp_start, sp_end),
                )
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = value

            start_ms = int(new_record[i][1]) * 1000
            end_ms = int(new_record[i][2]) * 1000
            try:
                seg_audio[start_ms:end_ms].export("audio/emotions.wav", format="wav")
                _out_prob, _score, _index, text_lab = classifier.classify_file("audio/emotions.wav")
                os.remove("audio/emotions.wav")
            except Exception:
                text_lab = ["None"]

            lab = (text_lab[0] if text_lab else "None") or "None"
            emo = emotion_dict.get(lab, None)
            records.append(
                [translated, final_sentence, new_record[i][1], new_record[i][2], speaker, emo]
            )
            print(translated, final_sentence, new_record[i][1], new_record[i][2], speaker, emo)

        os.environ["COQUI_TOS_AGREED"] = "1"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_cuda)

        shutil.rmtree("audio_chunks", ignore_errors=True)
        shutil.rmtree("su_audio_chunks", ignore_errors=True)
        os.makedirs("audio_chunks", exist_ok=True)
        os.makedirs("su_audio_chunks", exist_ok=True)

        natural_silence = records[0][2]
        previous_silence_time = 0.0
        if natural_silence >= 0.8:
            previous_silence_time = 0.8
            natural_silence -= 0.8
        else:
            previous_silence_time = natural_silence
            natural_silence = 0.0

        combined = AudioSegment.silent(duration=natural_silence * 1000)
        tip = 350

        def truncate_text(text: str, max_tokens: int = 50) -> str:
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return " ".join(words[:max_tokens]) + "..."

        for i in range(len(records)):
            print("previous_silence_time: ", previous_silence_time)
            tts.tts_to_file(
                text=truncate_text(records[i][0]),
                file_path=f"audio_chunks/{i}.wav",
                speaker_wav=f"speakers_audio/{records[i][4]}.wav",
                language=self.target_language,
                emotion=records[i][5],
                speed=2,
            )

            chunk = AudioSegment.from_file(f"audio_chunks/{i}.wav")
            chunk = chunk[: len(chunk) - tip]
            chunk.export(f"audio_chunks/{i}.wav", format="wav")

            lt = len(chunk) / 1000.0
            lo = max(records[i][3] - records[i][2], 0)
            theta = lo / lt if lt else 0.0
            input_file = f"audio_chunks/{i}.wav"
            output_file = f"su_audio_chunks/{i}.wav"

            # Boundaries match legacy: θ == 0.44 used the final else branch, not the silence-only branch.
            if theta < 1 and theta > 0.44:
                print("############################")
                theta_prim = (lo + previous_silence_time) / lt if lt else 1.0
                proc = subprocess.run(
                    f"ffmpeg -y -i {input_file} -filter:a 'atempo={1 / theta_prim}' -vn {output_file}",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    sc = lo + previous_silence_time
                    AudioSegment.silent(duration=sc * 1000).export(output_file, format="wav")
            elif theta < 0.44:
                AudioSegment.silent(duration=(lo + previous_silence_time) * 1000).export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=previous_silence_time * 1000)
                (silence + chunk).export(output_file, format="wav")

            seg_out = AudioSegment.from_file(output_file)
            lt = len(seg_out) / 1000.0
            lo = records[i][3] - records[i][2] + previous_silence_time
            if i + 1 < len(records):
                natural_silence = max(records[i + 1][2] - records[i][3], 0)
                if natural_silence >= 0.8:
                    previous_silence_time = 0.8
                    natural_silence -= 0.8
                else:
                    previous_silence_time = natural_silence
                    natural_silence = 0.0
                silence = AudioSegment.silent(duration=(max(lo - lt, 0) + natural_silence) * 1000)
                (seg_out + silence).export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=max(lo - lt, 0) * 1000)
                (seg_out + silence).export(output_file, format="wav")

            print("#######diff######: ", lo - lt)
            print("lo: ", lo)
            print("lt: ", lt)

        su_files = [f for f in os.listdir("su_audio_chunks") if f.endswith((".mp3", ".wav", ".ogg"))]
        su_files.sort(key=lambda x: int(x.split(".")[0]))
        for fname in su_files:
            combined += AudioSegment.from_file(os.path.join("su_audio_chunks", fname))

        total_length = len(AudioSegment.from_file(self.video_path)) / 1000.0
        combined += AudioSegment.silent(duration=abs(total_length - records[-1][3]) * 1000)
        combined.export("audio/output.wav", format="wav")

        separator = Separator()
        separator.load_model(model_filename="2_HP-UVR.pth")
        stem_path = separator.separate(self.video_path)[0]

        dub = AudioSegment.from_file("audio/output.wav")
        bg = AudioSegment.from_file(stem_path)
        dub.overlay(bg).export("audio/combined_audio.wav", format="wav")

        _run_ffmpeg_mux(self.video_path, "audio/combined_audio.wav", "output_video.mp4")
        shutil.move(stem_path, "audio/")

        if self.voice_denoising:
            _run_ffmpeg_mux(self.video_path, "audio/output.wav", "denoised_video.mp4")

        if self.lip_sync and self.voice_denoising:
            _run_wav2lip("denoised_video.mp4", "audio/output.wav")
        if self.lip_sync and not self.voice_denoising:
            _run_wav2lip("output_video.mp4", "audio/combined_audio.wav")

        if self.lip_sync and self.voice_denoising:
            shutil.move("Wav2Lip/results/result_voice.mp4", "results")
            if os.path.exists("output_video.mp4"):
                os.remove("output_video.mp4")
            shutil.move("denoised_video.mp4", "results")
        elif self.lip_sync and not self.voice_denoising:
            shutil.move("Wav2Lip/results/result_voice.mp4", "results")
            if os.path.exists("output_video.mp4"):
                os.remove("output_video.mp4")
        elif not self.lip_sync and self.voice_denoising:
            shutil.move("denoised_video.mp4", "results")
            if os.path.exists("output_video.mp4"):
                os.remove("output_video.mp4")
        else:
            shutil.move("output_video.mp4", "results")
