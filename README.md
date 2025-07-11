# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. It includes a workflow for
transcription and already provides a multimodal emotion classifier that
combines text and audio cues.

## Overview

EmotionKnowledge offers a CLI workflow for German speech transcription.
It can optionally run speaker diarization and store each utterance in a
database.  The package also provides a `MultimodalEmotionTagger` for
predicting emotions from text and the corresponding audio.

## Multimodal emotion tagging

`MultimodalEmotionTagger` combines a text classifier and a speech emotion
recognition model.  The class loads two Hugging Face pipelines – one for
`text-classification` and one for `audio-classification`.  By default it
uses the text model `oliverguhr/german-sentiment-bert` and the audio model
`superb/wav2vec2-base-superb-er`.  You can store the predicted emotion label
in your own data store, for example in a column named `Emotion_Text`.

An additional `AudioEmotionAnnotator` is available to annotate existing
utterances purely based on their audio clips. It now wraps an emotion model
and by default loads `padmalcom/wav2vec2-large-emotion-detection-german`.
The predicted label is stored in `audio_emotion_label` and the model's
confidence score in `audio_emotion_confidence`. The transcribed text is kept
in a separate field `audio_text`.

For text-only emotion detection the package offers `TextEmotionAnnotator`
which uses a German BERT model. It adds the fields `text_emotion_label` and
`text_emotion_confidence` to an entry while leaving any existing audio based
emotion information untouched.

## Default models

`MultimodalEmotionTagger` automatically loads
`oliverguhr/german-sentiment-bert` for text classification and
`superb/wav2vec2-base-superb-er` for speech emotion recognition.  The
`AudioEmotionAnnotator` uses `padmalcom/wav2vec2-large-emotion-detection-german`.
`TextEmotionAnnotator` defaults to `oliverguhr/german-sentiment-bert` for
classifying emotions in text.
The transcription workflow uses the open-source Whisper model (base size) to
convert German audio to text.

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
# store ``emotion`` alongside your text or audio entry
```

Add `--diarize` to enable speaker diarization with WhisperX using the
`WhisperXDiarizationWorkflow`.  When diarization is enabled you can also
store each speaker **utterance** in a local ChromaDB instance via
`SegmentSaver` by providing a database path and output directory for the
audio clips.

```bash
python -m emotion_knowledge path/to/audio.wav --diarize \
    --db-path mydb --clip-dir clips
```

Use `--whisperx-model` to choose the WhisperX model size when diarization is
enabled. The default is `medium`.

The script prints the resulting transcription to the console.

