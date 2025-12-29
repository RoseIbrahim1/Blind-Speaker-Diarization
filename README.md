# Blind-Speaker-Diarization

This project aims to design and implement a blind speaker diarization system that operates without any prior knowledge of the number of speakers in the audio recordings.

## Installation

```bash
pip install numpy soundfile librosa tensorflow resemblyzer spectralcluster

Run the Application
python interface.py

Input

Select an audio file in .wav format.

System Functionality

Predicts the number of speakers.

Performs speaker diarization.

Saves output files automatically.

Output

separated/: audio segments per speaker

concatenated/: full audio per speaker

Training Code

Training_Model.py: trains the speaker-counting model
