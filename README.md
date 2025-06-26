# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. The first implementation uses
text-only emotion classification so it can be easily extended later to
multimodal approaches.

## Overview

1. **AudioTranscriber** – transcribes German speech with WhisperX and can
    optionally perform speaker diarization.
2. **EmotionAnnotator** – labels each utterance using the lightweight
    `oliverguhr/german-sentiment-bert` model.
3. **EmotionTranscriptionPipeline** – chains the components to produce a list of
    annotated segments.
4. **TranscriptFormatter** – formats the annotated segments for display.

The new `emotion_transcription_pipeline()` function chains these Runnables using LangChain so you can process audio end-to-end:

```python
from emotion_knowledge.pipeline import emotion_transcription_pipeline

pipeline = emotion_transcription_pipeline()
print(pipeline.invoke("path/to/audio.wav"))
```

The code is structured so additional components can be inserted, such as
an audio-based emotion model.

## Default models

The built-in classes work with German data. `AudioTranscriber` loads the
WhisperX ASR model (size *small* by default) with `language='de'`. The
`EmotionAnnotator` relies on the `oliverguhr/german-sentiment-bert` model
to classify each utterance as positive, neutral or negative.

## Usage

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run a transcription:

```bash
python -m emotion_knowledge path/to/audio.wav
```

Add `--diarize` to enable speaker diarization with WhisperX. You can
choose a different WhisperX model size with `--model-size`:

```bash
python -m emotion_knowledge path/to/audio.wav --diarize --model-size medium
```

The script prints the resulting transcription to the console.

