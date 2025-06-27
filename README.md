# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. The first implementation uses
text-only emotion classification so it can be easily extended later to
multimodal approaches.

## Overview

1. **AudioTranscriber** – transcribes German speech with WhisperX and can
    optionally perform speaker diarization.
2. **SegmentDBWriter** – stores diarized segments in a Chroma DB and exports
    per-speaker audio clips.
3. **DBEmotionAnnotator** – fetches the stored segments one by one and annotates
    them using the lightweight `oliverguhr/german-sentiment-bert` model.
4. **EmotionTranscriptionPipeline** – chains the components to produce a list of
    annotated segments.
5. **TranscriptFormatter** – formats the annotated segments for display.

The new `emotion_transcription_pipeline()` function chains these Runnables using LangChain so you can process audio end-to-end:

```python
from emotion_knowledge.pipeline import emotion_transcription_pipeline

pipeline = emotion_transcription_pipeline()
print(pipeline.invoke("path/to/audio.wav"))
```

Each speaker segment is stored in a local Chroma database together with a
per-speaker audio clip. You can rerun only the emotion step later by
creating a `DBEmotionAnnotator` and calling it on the saved database.

The code is structured so additional components can be inserted, such as
an audio-based emotion model.

## Default models

The built-in classes work with German data. `AudioTranscriber` loads the
WhisperX ASR model (size *small* by default) with `language='de'`. The
`DBEmotionAnnotator` relies on the `oliverguhr/german-sentiment-bert` model
to classify each stored utterance as positive, neutral or negative. Long
utterances are truncated and GPU cache is cleared after each prediction to
keep memory usage low.

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

