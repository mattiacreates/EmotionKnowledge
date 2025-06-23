import argparse
from dataclasses import dataclass
from typing import Tuple

from transformers import pipeline


@dataclass
class AudioTranscriber:
    """Convert audio to text using a transformers ASR pipeline.

    The default model is ``openai/whisper-base`` for German speech
    recognition. The ``language`` option is passed when the pipeline is
    invoked rather than during initialisation.
    """

    model: str = "openai/whisper-base"

    def __post_init__(self) -> None:
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
        )

    def __call__(self, audio_path: str) -> str:
        result = self.pipeline(audio_path, generate_kwargs={"language": "de"})
        return result["text"].strip()


@dataclass
class TextEmotionAnnotator:
    """Annotate text with emotions using a Llama-based model.

    ``TextEmotionAnnotator`` prompts an instruction-tuned Llama model to
    summarise the emotion of the text. Only a single-word label is returned.
    The default model ``meta-llama/Meta-Llama-3-8B-Instruct`` is freely
    available from Hugging Face.
    """

    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __post_init__(self) -> None:
        self.pipeline = pipeline("text-generation", model=self.model)

    def __call__(self, text: str) -> str:
        prompt = (
            "Du bist ein Assistent, der die Emotion des folgenden Textes in einem Wort beschreibt. "
            "Gib nur dieses Wort aus.\n\nText: "
            + text
            + "\nEmotion:"
        )
        result = self.pipeline(prompt, max_new_tokens=3, do_sample=False)[0]["generated_text"]
        emotion = result.split("Emotion:")[-1].strip().split()[0]
        return f"[{emotion}] {text}"


@dataclass
class EmotionTranscriptionPipeline:
    """Run transcription then emotion annotation and return both strings."""

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
