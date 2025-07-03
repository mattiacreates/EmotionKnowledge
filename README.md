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

## Multimodal emotion tagging

`MultimodalEmotionTagger` combines a text classifier and a speech emotion
recognition model.  The class loads two Hugging Face pipelines – one for
`text-classification` and one for `audio-classification`.  By default it uses
models that understand German.  The predicted emotion label can be stored in a
database column named `Emotion_Text`.

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

To enrich the results with an emotion label based on both the transcribed text
and the original audio you can load ``MultimodalEmotionTagger`` in your own
notebook or script:

```python
from emotion_knowledge.emotion_tagger import MultimodalEmotionTagger
tagger = MultimodalEmotionTagger()
emotion = tagger.invoke(text, "path/to/audio.wav")
db["Emotion_Text"] = emotion
```

Add `--diarize` to enable speaker diarization with WhisperX. When diarization
is enabled you can also store each speaker **utterance** in a local ChromaDB
instance by providing a database path and output directory for the audio clips.

```bash
python -m emotion_knowledge path/to/audio.wav --diarize \
    --db-path mydb --clip-dir clips
```

Use `--whisperx-model` to choose the WhisperX model size when diarization is
enabled. The default is `medium`.

The script prints the resulting transcription to the console.

