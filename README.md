# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. The first implementation uses
text-only emotion classification so it can be easily extended later to
multimodal approaches.

## Overview

1. **AudioTranscriber** – converts an audio file to text using the
   Hugging Face `transformers` ASR pipeline (Whisper).
2. **TextEmotionAnnotator** – detects emotions in the text using a
   transformers text-classification model.
3. **EmotionTranscriptionPipeline** – orchestrates the two steps. It
   returns both the plain transcription and the emotion-enriched text.

The code is structured so additional components can be inserted, such as
an audio-based emotion model.

## Default models

The built-in classes work with German data. `AudioTranscriber` loads
`openai/whisper-base` with `language='de'` for speech recognition, while
`TextEmotionAnnotator` uses `oliverguhr/german-emotion-bert` for emotion
classification.

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

## Testing

Run tests with:

```bash
pytest
```

Tests use lightweight stubs so they run without downloading heavy
models. A tiny `transformers` package in `transformers/` provides the
stub. Remove or bypass this directory when running the example with the
real `transformers` library so the actual models are loaded.
