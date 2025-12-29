import numpy as np
import soundfile as sf
import librosa
import math
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                    Flatten, Dense, Dropout, LSTM, TimeDistributed)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Configuration
tf.keras.backend.set_floatx('float32')
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
DURATION = 10
FRAME_LENGTH = int(SAMPLE_RATE * DURATION)
N_MELS = 64

def load_audio(path):
    audio, sr = sf.read(path)
    audio = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(audio) < FRAME_LENGTH:
        audio = np.pad(audio, (0, FRAME_LENGTH - len(audio)), 'constant')
    return audio[:FRAME_LENGTH].astype('float32')

def create_mel_spectrogram(audio):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2,
        fmax=8000
    )
    spec_db = librosa.power_to_db(mel_spec, top_db=80)  
    return spec_db.T.astype('float32')

class AudioGenerator(Sequence):
    def __init__(self, file_list, batch_size=32, shuffle=True):
        self.file_list = file_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_files = self.file_list[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.zeros((len(batch_files), *SPEC_SHAPE, 1))
        y = np.zeros((len(batch_files), 5))
        
        for i, (path, label) in enumerate(batch_files):
            try:
                audio = load_audio(path)
                spec = create_mel_spectrogram(audio)
                X[i] = spec[..., np.newaxis]  # Add channel dimension
                y[i] = to_categorical(int(label)-1, num_classes=5)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
                
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_list)

def build_model(input_shape):
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(*input_shape, 1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        
        TimeDistributed(Flatten()),
        
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
    
    model.compile(
    # Load data list
        optimizer=Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    import glob
    global SPEC_SHAPE

    # Create checkpoint directory
    os.makedirs("checkpoints_mixed_Ten_tryy", exist_ok=True)

    # Set to True to resume training from last checkpoint, False to start from scratch
    RESUME_TRAINING = True

    with open(r"train_list_try.txt", "r") as f:
        file_list = [line.strip().split(",") for line in f if line.strip()]

    # Determine input shape
    test_audio = np.zeros(FRAME_LENGTH)
    test_spec = create_mel_spectrogram(test_audio)
    SPEC_SHAPE = test_spec.shape
    print(f"Input shape: {SPEC_SHAPE}")

    # Split into training and validation sets
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    # Create generators
    train_gen = AudioGenerator(train_files)
    val_gen = AudioGenerator(val_files, shuffle=False)

    # Build model
    model = build_model(SPEC_SHAPE)
    model.summary()

    # Load latest checkpoint if resuming
    initial_epoch = 0
    if RESUME_TRAINING:
        checkpoint_files = sorted(glob.glob("checkpoints_mixed_Ten_tryy/epoch_*.h5"), 
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
            print(f"\nLoading weights from {latest_checkpoint}")
            print(f"Resuming training from epoch {initial_epoch + 1}\n")
            model.load_weights(latest_checkpoint)
        else:
            print("No checkpoint found. Starting from scratch.")
    print(f"Current learning rate: {model.optimizer.learning_rate.numpy():.6f}")

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints_mixed_Ten_tryy/epoch_{epoch:02d}.h5',
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    # Save final model
    model.save("speaker_model_fixed.h5")
    print(f"Training completed. Best val accuracy: {max(history.history['val_accuracy']):.2f}")

if __name__ == "__main__":
    main()
