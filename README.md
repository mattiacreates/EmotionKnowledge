# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. The first implementation uses
text-only emotion classification so it can be easily extended later to
multimodal approaches.

## Overview

1. **AudioTranscriber** – converts an audio file to text using the
   Hugging Face `transformers` ASR pipeline (Whisper).
2. **TextEmotionAnnotator** – prompts a Llama model to label the emotion
   of the text.
3. **EmotionTranscriptionPipeline** – orchestrates the two steps. It
   returns both the plain transcription and the emotion-enriched text.

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

Run the example:

```bash
python -m emotion_knowledge path/to/audio.wav
```

The script prints the plain transcription and the transcription with
emotion labels.

