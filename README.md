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

For speaker diarization the project now leverages NVIDIA's **Sortformer
Diarizer 4spk v1**. Sortformer resolves the permutation problem by ordering
speech segments by their arrival time and consists of an 18-layer NeMo Encoder
for Speech Tasks (NEST) followed by an 18-layer Transformer with hidden size
192 and four sigmoid outputs per frame. The model operates on 16 kHz mono audio
and can distinguish up to four speakers.

### Installing NeMo and Sortformer

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### Loading the model

```python
from nemo.collections.asr.models import SortformerEncLabelModel

diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
diar_model.eval()
predicted_segments = diar_model.diarize(audio="/path/to/audio.wav", batch_size=1)
```

The model works offline and was trained primarily on English speech; performance
may degrade on more speakers or out-of-domain data.

## Usage

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run a transcription:

```bash
python -m emotion_knowledge path/to/audio.wav
```

`SortformerDiarizationWorkflow`.  When diarization is enabled you can also
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

When diarization is enabled you can specify a Sortformer checkpoint via
`--sortformer-model`. By default the workflow loads the pretrained
`nvidia/diar_sortformer_4spk-v1` model, but you can also pass the path to a
downloaded `.nemo` file.

```bash
python -m emotion_knowledge path/to/audio.wav --diarize --sortformer-model /path/to/diar_sortformer_4spk-v1.nemo
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
%pip install langchain transformers openai-whisper
# install NeMo for Sortformer diarization
%pip install Cython packaging
%pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]

%cd EmotionKnowledge
!python -m emotion_knowledge /path/to/audio.wav --diarize \
    --sortformer-model nvidia/diar_sortformer_4spk-v1
```

Replace `/path/to/audio.wav` with a file from your Google Drive or an uploaded
sample. The results are printed to the notebook and, when diarization is
enabled, stored in the local `segment_db` directory.

