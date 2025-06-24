import argparse
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from typing import Tuple

import whisper
from transformers import pipeline


@tool
def transcribe_audio_whisper(audio_path: str) -> str:
    """Transcribe German audio using Whisper (local openai/whisper-base)."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="de")
    return result["text"].strip()


@tool
def annotate_emotion_llama(text: str) -> str:
    """Annotate a single emotion word to the text using a LLaMA instruction model."""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    generator = pipeline("text-generation", model=model_id)
    prompt = (
        "Du bist ein Assistent, der die Emotion des folgenden Textes in einem Wort beschreibt. "
        "Gib nur dieses Wort aus.\n\nText: "
        + text
        + "\nEmotion:"
    )
    result = generator(prompt, max_new_tokens=3, do_sample=False)[0]["generated_text"]
    emotion = result.split("Emotion:")[-1].strip().split()[0]
    return f"[{emotion}] {text}"


class EmotionTranscriptionWorkflow(Runnable):
    """A LangChain-compatible workflow that transcribes audio and annotates it with emotion."""

    def invoke(self, audio_path: str) -> Tuple[str, str]:
        raw_text = transcribe_audio_whisper.invoke(audio_path)
        annotated = annotate_emotion_llama.invoke(raw_text)
        return raw_text, annotated


def main():
    parser = argparse.ArgumentParser(description="German audio transcription with emotion annotation.")
    parser.add_argument("audio", help="Path to audio file (WAV/MP3)")
    args = parser.parse_args()

    workflow = EmotionTranscriptionWorkflow()
    raw, emotion = workflow.invoke(args.audio)

    print("📄 Plain Transcription:\n", raw)
    print("\n💬 Emotion Annotated:\n", emotion)


if __name__ == "__main__":
    main()
