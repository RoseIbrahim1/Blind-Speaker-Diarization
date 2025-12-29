1. Install required packages:
   pip install numpy soundfile librosa tensorflow resemblyzer spectralcluster

2. Run the Interface script:
   python interface.py

3. Select an audio file (.wav).

4. The program will:
   - Predict the number of speakers in the audio.
   - Perform speaker diarization and save output files.

4. Output files saved in the same folder as the audio:
   - 'separated' folder: segments of each speaker.
   - 'concanated' folder: concatenated audio per speaker.
_________________________________________________________________________

In Training code Folder:

Training_Model.py is the code to train the model for speaker counting