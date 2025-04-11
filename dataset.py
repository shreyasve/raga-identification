import os
import numpy as np
import librosa
import scipy.signal as signal
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def compute_dft(audio_path, max_length=200):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    N = len(y)
    dft = np.abs(fft(y))[:N // 2]  # Take positive frequencies
    dft = np.log1p(dft)  # Log-scale to match human perception
    
    # Apply Savitzky-Golay filter to smooth vibrato
    dft_smooth = signal.savgol_filter(dft, window_length=11, polyorder=2)
    
    # Normalize frequencies
    dft_smooth = (dft_smooth - np.min(dft_smooth)) / (np.max(dft_smooth) - np.min(dft_smooth))
    
    if len(dft_smooth) > max_length:
        dft_smooth = dft_smooth[:max_length]
    else:
        padding = max_length - len(dft_smooth)
        dft_smooth = np.pad(dft_smooth, (0, padding), mode='constant')
    
    return dft_smooth

def load_dataset(folder_path, max_length=200):
    X, y = [], []
    label_to_index = {}
    
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            audio_path = os.path.join(folder_path, file)
            label = file.split('_')[0]  # Assuming label is first part of filename
            dft_smooth = compute_dft(audio_path, max_length)
            X.append(dft_smooth)
            
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
            y.append(label_to_index[label])
    
    return np.array(X), np.array(y), label_to_index

def main():
    folder_path = "C:\\Users\\shrey\\OneDrive\\Desktop\\raga_detection\\new_raga\\ragas"
    X, y, label_to_index = load_dataset(folder_path, max_length=200)
    
    X = X[..., np.newaxis]  # Reshape for CNN input
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(200, 1), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(len(label_to_index), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32, class_weight=class_weights, callbacks=[early_stopping, lr_scheduler])
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}")
    
    def predict_raga(audio_path):
        dft_smooth = compute_dft(audio_path, max_length=200)
        dft_smooth = dft_smooth[np.newaxis, ..., np.newaxis]
        prediction = model.predict(dft_smooth)
        predicted_index = np.argmax(prediction)
        return list(label_to_index.keys())[list(label_to_index.values()).index(predicted_index)]
    
    sample_audio = "C:\\Users\\shrey\\OneDrive\\Desktop\\raga_detection\\new_raga\\ragas\\asavari25.wav"
    predicted_raga = predict_raga(sample_audio)
    print(f"Predicted Raga: {predicted_raga}")
    
if __name__ == '__main__':
    main()
