# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. The first implementation uses
text-only emotion classification so it can be easily extended later to
multimodal approaches.

## Overview

1. **AudioTranscriber** – converts an audio file to text using the
   Hugging Face `transformers` ASR pipeline (Whisper).
2. **TextEmotionAnnotator** – prompts a Llama model to label the emotion of the text.
3. **EmotionTranscriptionPipeline** – orchestrates the two steps. It returns both the plain transcription and the emotion-enriched text.
4. **EmotionDetector** – uses a multimodal Hugging Face model to detect the emotion of each diarized segment.
5. **TranscriptFormatter** – formats the annotated segments for display.

The new `emotion_transcription_pipeline()` function chains these Runnables using LangChain so you can process audio end-to-end:

```python
from emotion_knowledge.pipeline import emotion_transcription_pipeline

pipeline = emotion_transcription_pipeline()
print(pipeline.invoke("path/to/audio.wav"))
```

The code is structured so additional components can be inserted, such as
an audio-based emotion model.

## Default models

The built-in classes work with German data. `AudioTranscriber` uses
`openai/whisper-base` for speech recognition and passes
`language='de'` when invoking the pipeline. `TextEmotionAnnotator` uses
``meta-llama/Meta-Llama-3-8B-Instruct`` to generate a single emotion
label.

## Usage

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run a transcription:

```bash
python -m emotion_knowledge path/to/audio.wav
```

Add `--diarize` to enable speaker diarization with WhisperX:

```bash
python -m emotion_knowledge path/to/audio.wav --diarize
```

The script prints the resulting transcription to the console.

