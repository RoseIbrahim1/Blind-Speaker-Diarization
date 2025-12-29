from diarNS import run_diarization
from mypredict_imp import predict_speaker_count
import tkinter as tk
from tkinter import filedialog, messagebox

def main():
    # Create a hidden Tkinter root window for the file dialog
    root = tk.Tk()
    root.withdraw()

    # Ask user to select an audio file
    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("Audio Files", "*.wav *.flac *.mp3")]
    )

    if not file_path:
        print("No file selected. Exiting...")
        return

    try:
        # Predict speaker count
        print("Predicting speaker count...")
        num_speakers = predict_speaker_count(file_path)
        print(f"Predicted speaker count: {num_speakers}")

        # Run diarization
        print("Running diarization...")
        run_diarization(num_speakers, file_path)
        print("Diarization completed.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()