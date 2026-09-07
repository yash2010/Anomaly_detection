import numpy as np
import os

try:
    print("Loading inputX_compressed.npz...")
    data = np.load('inputX_compressed.npz')
    if 'X' in data:
        X = data['X']
        print(f"X shape: {X.shape}")
        print(f"X min: {X.min()}, X max: {X.max()}, X mean: {X.mean()}")
        print(f"X dtype: {X.dtype}")
    else:
        print("Key 'X' not found in inputX_compressed.npz")
        print(f"Keys found: {list(data.keys())}")

    print("\nLoading labelX.npy...")
    y = np.load('labelX.npy')
    print(f"y shape: {y.shape}")
    print(f"y unique values: {np.unique(y)}")
    
except Exception as e:
    print(f"Error: {e}")
