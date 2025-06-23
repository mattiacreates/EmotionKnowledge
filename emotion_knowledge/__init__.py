import argparse
from dataclasses import dataclass
from typing import Tuple

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - fallback when transformers isn't installed
    from transformers_stub import pipeline


@dataclass
class AudioTranscriber:
    """Convert audio to text using a transformers ASR pipeline."""

    model: str = "openai/whisper-base"

    def __post_init__(self) -> None:
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            language="de",
            task="transcribe",
        )

    def __call__(self, audio_path: str) -> str:
        result = self.pipeline(audio_path)
        return result["text"].strip()


@dataclass
class TextEmotionAnnotator:
    """Annotate text with emotions using a transformers classifier.

    Long inputs are truncated to the model's maximum length when the
    underlying pipeline is called.
    """

    model: str = "oliverguhr/german-emotion-bert"

    def __post_init__(self) -> None:
        self.pipeline = pipeline("text-classification", model=self.model, return_all_scores=True)

    def __call__(self, text: str) -> str:
        if hasattr(self.pipeline, "tokenizer"):
            scores = self.pipeline(
                text,
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            )[0]
        else:
            scores = self.pipeline(text, truncation=True)[0]
        top = max(scores, key=lambda s: s["score"])
        return f"[{top['label']}] {text}"


@dataclass
class EmotionTranscriptionPipeline:
    transcriber: AudioTranscriber
    annotator: TextEmotionAnnotator

    def __call__(self, audio_path: str) -> Tuple[str, str]:
        text = self.transcriber(audio_path)
        annotated = self.annotator(text)
        return text, annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio and annotate emotions")
    parser.add_argument("audio", help="Path to an audio file")
    args = parser.parse_args()

    pipeline = EmotionTranscriptionPipeline(AudioTranscriber(), TextEmotionAnnotator())
    text, annotated = pipeline(args.audio)
    print("Plain transcription:\n", text)
    print("\nWith emotion:\n", annotated)


if __name__ == "__main__":
    main()
