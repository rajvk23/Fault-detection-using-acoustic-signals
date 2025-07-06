import os
import numpy as np
import pandas as pd
import pywt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "AirCompressor_Data"
SAMPLE_RATE = 50000
SAMPLES = 250000

def butter_filter(signal, lowcut=400, highcut=12000, fs=50000, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def clip_signal(signal, segment_duration=1.0, overlap=0.5):
    segment_len = int(segment_duration * SAMPLE_RATE)
    step = int(segment_len * (1 - overlap))
    min_std = float('inf')
    best_segment = signal[:segment_len]
    for i in range(0, len(signal) - segment_len + 1, step):
        segment = signal[i:i+segment_len]
        std = np.std(segment)
        if std < min_std:
            min_std = std
            best_segment = segment
    return best_segment

def smooth_signal(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std

def extract_features(signal):
    features = []

    # Time-domain features
    features.append(np.mean(np.abs(signal)))
    features.append(np.max(signal))
    features.append(np.sqrt(np.mean(signal**2))) 
    features.append(np.var(signal))
    features.append(pd.Series(signal).kurt())
    features.append(np.max(signal)/np.sqrt(np.mean(signal**2)))
    features.append(np.sqrt(np.mean(signal**2))/np.mean(np.abs(signal))) 
    features.append(pd.Series(signal).skew())

    # Frequency-domain (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_energy = fft_vals**2
    bins = np.array_split(fft_energy, 8)
    for b in bins:
        features.append(np.sum(b)/np.sum(fft_energy))

    # DWT 
    coeffs = pywt.wavedec(signal, 'db4', level=6)
    for c in coeffs[1:4]: 
        features.append(np.var(c))

    return features

def process_file(filepath):
    signal = np.fromfile(filepath, dtype=np.float32)
    signal = butter_filter(signal)
    signal = clip_signal(signal)
    signal = smooth_signal(signal)
    signal = normalize_signal(signal)
    features = extract_features(signal)
    return features

def main():
    all_features = []
    all_labels = []
    
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.endswith('.dat'):
                filepath = os.path.join(label_path, file)
                features = process_file(filepath)
                all_features.append(features)
                all_labels.append(label)
                print(f"Processed {file} -> Label: {label}")

    df = pd.DataFrame(all_features)
    df['label'] = all_labels
    df.to_csv("features_18.csv", index=False)
    print("✅ Features saved to features_18.csv")

if __name__ == "__main__":
    main()
