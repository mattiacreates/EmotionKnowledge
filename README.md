# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. The first implementation uses
text-only emotion classification so it can be easily extended later to
multimodal approaches.

## Overview

1. **AudioTranscriber** – converts an audio file to text with Whisper or
   WhisperX (German, optional diarization). Transcripts are cached on
   disk.
2. **SegmentDBWriter** – writes each utterance into a ChromaDB
   collection and exports individual WAV clips.
3. **DBEmotionAnnotator** – batches the texts in the database and runs a
   German emotion classification model. Results are stored back in the
   DB.
4. **TranscriptFormatter** – formats the annotated segments as readable
   strings.
5. **emotion_transcription_pipeline** – combines all steps using
   LangChain's ``RunnableSequence``.

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

Use `--diarize` for speaker labels and `--load-in-8bit` to reduce GPU
memory usage. All settings can be configured via command line. Pass
`--emotion-model` to choose a different HuggingFace classifier:

```bash
python -m emotion_knowledge path/to/audio.wav --diarize --batch-size 16 \
    --db-path mydb --clip-dir clips
```

```bash
python -m emotion_knowledge path/to/audio.wav --emotion-model arpanghoshal/EmoRoBERTa
```

The script prints the emotion-enriched transcript to the console.

