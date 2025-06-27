import json
import os
from pathlib import Path
from typing import Dict, List

import torch


class AudioTranscriber:
    """Transcribe German audio with optional speaker diarization."""

    def __init__(self, model_size: str = "base", diarize: bool = False, cache_dir: str = "transcripts"):
        self.model_size = model_size
        self.diarize = diarize
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, audio_path: str) -> Dict:
        audio_path = str(audio_path)
        cache_file = self.cache_dir / (Path(audio_path).stem + ".json")
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        import whisperx
        result = {}
        model = whisperx.load_model(self.model_size, device=self.device, language="de")
        transcription = model.transcribe(audio_path)
        segments = transcription.get("segments", [])

        if self.diarize:
            token = os.getenv("HF_TOKEN")
            diarize_model = whisperx.DiarizationPipeline(device=self.device, use_auth_token=token)
            diarize_segments = diarize_model(audio_path)
            segments = whisperx.assign_word_speakers(diarize_segments, transcription)["segments"]

        result["audio_path"] = audio_path
        result["segments"] = [
            {
                "text": s.get("text", ""),
                "speaker": s.get("speaker", "Speaker"),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
            }
            for s in segments
        ]

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        torch.cuda.empty_cache()
        return result
