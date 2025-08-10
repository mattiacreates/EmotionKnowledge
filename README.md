# Emotion Knowledge

This repository provides a minimal example of turning audio into text
and annotating that text with emotions. It includes a workflow for
transcription and provides utilities to annotate text or audio with emotions.

## Overview

EmotionKnowledge offers a CLI workflow for German speech transcription.
It can optionally run speaker diarization and store each utterance in a database.
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

`WhisperXDiarizationWorkflow`.  When diarization is enabled you can also
store each speaker **utterance** in a local ChromaDB instance via
`SegmentSaver` by providing a database path and output directory for the
audio clips.

When grouping words into utterances you can enable sentence-level grouping by
merging consecutive utterances from the same speaker. The helper function
`_group_utterances` now accepts `merge_sentences=True` to perform this merge and
the diarization workflow uses this option by default so each saved clip contains
full sentences per speaker.

```bash
python -m emotion_knowledge path/to/audio.wav --diarize \
    --db-path mydb --clip-dir clips
```

Use `--whisperx-model` to choose the WhisperX model size when diarization is
enabled. The default is `medium`, but you can also select `base`, `small`, or
`large` depending on your resource constraints. The workflow automatically
selects the compute type: `float16` when a CUDA-enabled GPU is available and
`int8` otherwise, logging the chosen precision for transparency.

Short backchannel interjections are preserved as separate utterances. Pass
`--no-preserve-backchannels` to merge them back into the surrounding speech. Use
`--preserve-end-times` to keep original end timestamps (enabled by default).

For example, the following command uses the smaller `base` model:

```bash
python -m emotion_knowledge path/to/audio.wav --diarize --whisperx-model base
```

The script prints the resulting transcription to the console.

## Running on Google Colab

You can also run the workflow on [Google Colab](https://colab.research.google.com/)
with a GPU runtime. The following snippet installs the required libraries,
clones the repository and transcribes an audio file:

```python
# start in a clean workspace
%cd /content
!rm -rf EmotionKnowledge

# install CUDA compatible PyTorch
%pip install --upgrade pip
%pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118

# clone this repo and install the dependencies
!git clone https://github.com/mattiacreates/EmotionKnowledge.git
%pip install -r EmotionKnowledge/requirements.txt

# optional: extra models used by the annotators
%pip install langchain transformers

%cd EmotionKnowledge
!python -m emotion_knowledge /path/to/audio.wav --diarize \
    --whisperx-model medium
```

Replace `/path/to/audio.wav` with a file from your Google Drive or an uploaded
sample. The results are printed to the notebook and, when diarization is
enabled, stored in the local `segment_db` directory.

