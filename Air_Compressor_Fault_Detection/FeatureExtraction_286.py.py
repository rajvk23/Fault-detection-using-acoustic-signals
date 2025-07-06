import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt
from scipy.stats import entropy

DATA_DIR = "AirCompressor_Data"
SAMPLE_RATE = 50000
SAMPLES = 250000

def butter_filter(signal, lowcut=400, highcut=12000, fs=SAMPLE_RATE, order=9):
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
    p_low, p_high = np.percentile(signal, [0.025, 99.975])
    return 2 * (signal - p_low) / (p_high - p_low) - 1

# 1. Time Domain Features (8 features) 
def extract_time_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    abs_mean = np.mean(np.abs(signal))
    crest_factor = np.max(np.abs(signal)) / rms if rms != 0 else 0
    shape_factor = rms / abs_mean if abs_mean != 0 else 0
    
    return [
        abs_mean,
        np.max(np.abs(signal)),
        rms,
        np.var(signal),
        pd.Series(signal).kurt(),
        crest_factor,
        shape_factor,
        pd.Series(signal).skew()
    ]

# 2. Frequency Domain (FFT) Features (8 features) 
def extract_fft_features(signal):
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_energy = fft_vals**2
    total_energy = np.sum(fft_energy)
    if total_energy == 0:
        return [0.0] * 8
    bins = np.array_split(fft_energy, 8)
    return [np.sum(b) / total_energy for b in bins]

# 3. Discrete Wavelet Transform (DWT) Features (9 features) 
def extract_dwt_features(signal, wavelet='db4', level=6):
    coeffs = pywt.wavedec(signal, wavelet, level=level)    
    features = []
    features.extend([np.var(coeffs[-i]) for i in range(1, 4)]) 
    for i in range(4, 7): 
        detail_coeff = coeffs[-i]
        autocorr = np.correlate(detail_coeff, detail_coeff, mode='full')
        features.append(np.var(autocorr))

    for i in range(1, 4):
        smoothed_coeff = np.convolve(coeffs[-i], np.ones(5)/5, mode='valid')
        features.append(np.mean(smoothed_coeff))
        
    return features

# 4. Wavelet Packet Transform (WPT) Features (254 features) 
def extract_wpt_features(signal, wavelet='db4', maxlevel=7):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    features = []
    for level in range(1, maxlevel + 1):
        nodes = [node.path for node in wp.get_level(level, 'natural')]
        level_features = [np.sum(np.square(wp[node].data)) for node in nodes]
        features.extend(level_features)
    return features

# 5. Morlet Wavelet Transform (MWT) Features (7 features) 
def _calculate_entropy(data):
    value_counts = pd.Series(data).value_counts(normalize=True, sort=False)
    return entropy(value_counts.values)

def extract_morlet_features(signal, wavelet='cmor1.5-1.0'):
    widths = np.arange(1, 32) 
    cwt_coeffs, _ = pywt.cwt(signal, widths, wavelet)
    transformed_signal = np.abs(cwt_coeffs).flatten()

    features = []
    
    # 1. Wavelet Entropy
    features.append(_calculate_entropy(transformed_signal))
    
    # 2. Sum of Peaks (using a simple definition)
    peak_threshold = np.mean(transformed_signal) + 2 * np.std(transformed_signal)
    features.append(np.sum(transformed_signal[transformed_signal > peak_threshold]))
    
    # 3. Standard Deviation
    features.append(np.std(transformed_signal))
    
    # 4. Kurtosis
    features.append(pd.Series(transformed_signal).kurt())
    
    # 5. Zero Crossing Rate (on the original signal, as it's a time-domain property)
    features.append(np.sum(np.diff(np.sign(signal)) != 0) / (len(signal)-1))

    # 6. Variance
    features.append(np.var(transformed_signal))
    
    # 7. Skewness
    features.append(pd.Series(transformed_signal).skew())
    
    return features


def extract_all_features(signal):
    f_time = extract_time_features(signal)
    f_fft = extract_fft_features(signal)
    f_dwt = extract_dwt_features(signal)
    f_wpt = extract_wpt_features(signal)
    f_mwt = extract_morlet_features(signal)
    
    feature_vector = f_time + f_fft + f_dwt + f_wpt + f_mwt
    return feature_vector

def process_file(filepath):
    try:
        signal = np.fromfile(filepath, dtype=np.float32)
        if signal.size < SAMPLES:
            signal = np.pad(signal, (0, SAMPLES - signal.size), 'constant')
        signal = signal[:SAMPLES]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    signal = butter_filter(signal)
    signal = clip_signal(signal)
    signal = smooth_signal(signal)
    signal = normalize_signal(signal)
    return extract_all_features(signal)

def main():
    all_features_list = []
    all_labels = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return
        
    for label in sorted(os.listdir(DATA_DIR)): 
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
            
        print(f"\nProcessing category: {label}...")
        for filename in sorted(os.listdir(label_path)):
            if filename.endswith('.dat'):
                filepath = os.path.join(label_path, filename)
                features = process_file(filepath)
                if features:
                    all_features_list.append(features)
                    all_labels.append(label)
                    print(f"  > Processed {filename}")

    if not all_features_list:
        print("No .dat files were processed. Exiting.")
        return

    feature_names = [f'TD_{i+1}' for i in range(8)] + \
                    [f'FFT_{i+1}' for i in range(8)] + \
                    [f'DWT_{i+1}' for i in range(9)] + \
                    [f'WPT_{i+1}' for i in range(254)] + \
                    [f'MWT_{i+1}' for i in range(7)]

    df = pd.DataFrame(all_features_list, columns=feature_names)
    df['label'] = all_labels
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True) 
    df.to_csv("features_286.csv", index=False)
    print("\n" + "="*50)
    print(f"✅ All {len(all_features_list)} files processed.")
    print(f"✅ Feature matrix shape: {df.shape}")
    print("✅ All features extracted and saved to 'features_286.csv'")
    print("="*50)


if __name__ == "__main__":
    main()