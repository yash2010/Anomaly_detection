from __future__ import annotations

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from config import dataConfig

def load_fft_data(cfg: dataConfig):
    """
    Loads FFT-transformed data from a saved .npy file and filters out IDs
    that have fewer than the required number of samples.

    Args:
        file_path (str): Path to the .npy file containing FFT data.
        min_samples (int): Minimum number of samples required per ID.

    Returns:
        dict: Filtered FFT data organized by folder name.
    """
        
    data = np.load(cfg.file_path, allow_pickle=True).item()
    
    id_counts = Counter()
    for folder, data_list in data.items():
        for label, (freq, fft_mag) in data_list:
            match = re.search(r'[\\/](\d+)_', label)
            if match:
                id_counts[match.group(1)] += 1


    print("Samples per ID:")
    for id_, count in id_counts.items():
        print(f"ID {id_}: {count} samples")


    exclude_few_samples = [id_ for id_, count in id_counts.items() if count < cfg.min_samples]
    print("\nIDs to exclude (few samples):", exclude_few_samples)


    sorted_ids = sorted(id_counts.keys(), key=lambda x: int(x))
    sorted_counts = [id_counts[id_] for id_ in sorted_ids]
    colors = ['red' if id_ in exclude_few_samples else 'blue' for id_ in sorted_ids]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_ids, sorted_counts, color=colors)
    plt.xticks(rotation=90)
    plt.xlabel("ID")
    plt.ylabel("Number of samples")
    plt.title("Number of Samples per ID (Red = Too Few)")
    plt.tight_layout()
    # plt.show() # Commented out for non-interactive environments

    for folder, data_list in data.items():
        data[folder] = [
            (label, (freq, fft_mag))
            for label, (freq, fft_mag) in data_list
            if not (match := re.search(r'[\\/](\d+)_', label)) or match.group(1) not in exclude_few_samples
        ]

    print("\nData filtered. Remaining samples per folder:")
    for folder, data_list in data.items():
        print(f"{folder}: {len(data_list)} samples")

    return data


def preprocessing(data, cfg: dataConfig):
    """
    Truncates FFT signals to a max frequency and length, then scales them.

    Args:
        data (dict): FFT data dictionary from load_fft_data().
        max_len (int): Number of FFT bins to keep.
        max_freq (int): Frequency cutoff in Hz.

    Returns:
        X (np.ndarray): Original FFT magnitude data (unscaled).
        y (np.ndarray): Corresponding folder labels.
        Xscaled (np.ndarray): Standard-scaled FFT magnitude data.
    """

    all_samples = []
    fft_labels = []

    for folder, data_list in data.items():
        for label, (freq, fft_mag) in data_list:
            mask = freq <= cfg.max_freq
            fft_mag = fft_mag[mask]

            if len(fft_mag) < cfg.max_len:
                fft_mag = np.pad(fft_mag, (0, cfg.max_len - len(fft_mag)), 'constant')
            else:
                fft_mag = fft_mag[:cfg.max_len]
            
            all_samples.append(fft_mag)
            fft_labels.append(folder)
    
    X = np.array(all_samples)
    y = np.array(fft_labels)

    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(X)
    
    # plt.figure(figsize=(14, 5))
    # plt.plot(X[1], label = 'Original', linewidth = 2)
    # plt.plot(Xscaled[1], label = 'Scaled signal', linewidth = 2, linestyle = '--')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return X, y, Xscaled, scaler
