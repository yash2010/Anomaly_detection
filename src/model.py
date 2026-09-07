from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, UpSampling1D,
                                     Activation, BatchNormalization, Flatten,
                                     Dense, Reshape, Concatenate)
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
import numpy as np
from config import modelConfig, clusterConfig

def autoencoder(cfg: modelConfig):
    """
    Constructs a 1D Convolutional Autoencoder for feature extraction.

    Args:
        cfg.input_dim (int): Length of input FFT signal.
        cfg.bottleneck_dim (int): Size of latent space.
        cfg.lr (float): Learning rate for Adam optimizer.
        cfg.filters (list): List of filter sizes for the encoder layers. 
                        Decoder will use the reverse.

    Returns:
        model (tf.keras.Model): Compiled autoencoder model.
    """
        
    input_layer = Input(shape=(cfg.input_dim, 1))
    x = input_layer
    
    # Check input dimension compatibility
    num_layers = len(cfg.filters)
    if cfg.input_dim % (2 ** num_layers) != 0:
        print(f"Warning: Input dimension {cfg.input_dim} is not divisible by {2**num_layers}. "
              "Autoencoder shape mismatch might occur in the decoder.")
    
    # Encoder
    skip_connections = []
    for f in cfg.filters:
        x = Conv1D(f, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(f, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        skip_connections.append(x)
        x = MaxPooling1D(2, padding='same')(x)

    p_x = x
    x = Flatten()(p_x) 
    bottle_neck = Dense(cfg.bottleneck_dim, activation = 'linear', name='bottle_neck')(x)

    # Decoder
    x = Dense(p_x.shape[1] * p_x.shape[2], activation='relu')(bottle_neck)
    x = Reshape((p_x.shape[1], p_x.shape[2]))(x)

    for i, f in enumerate(reversed(cfg.filters)):
        x = UpSampling1D(2)(x)
        # Handle shape mismatch if any due to padding
        # In this specific architecture with powers of 2, it should be fine usually
        # But for robustness, we might need cropping or padding. 
        # For now, we assume cfg.input_dim is power of 2 compatible.
        
        # Concatenate with skip connection
        # skip_connections is [c2, c3, c4, c5] (using original names)
        # reversed(cfg.filters) corresponds to processing c5, then c4, etc.
        # So we need skip_connections[-(i+1)]
        
        skip_conn = skip_connections[-(i+1)]
        
        # Resize x to match skip_conn if needed (simple check)
        if x.shape[1] > skip_conn.shape[1]:
            x = tf.keras.layers.Cropping1D(cropping=(0, x.shape[1] - skip_conn.shape[1]))(x)
        elif x.shape[1] < skip_conn.shape[1]:
            x = tf.keras.layers.ZeroPadding1D(padding=(0, skip_conn.shape[1] - x.shape[1]))(x)

        x = Concatenate()([x, skip_conn])
        
        x = Conv1D(f, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(f, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    output_layer = Conv1D(1, 3, activation='linear', padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(cfg.lr), loss='mse')
    # model.summary()

    return model 


def get_latent_features(model, X_reshaped, batch_size):
    """
    Extracts the encoder portion from the full autoencoder model.

    Args:
        model (tf.keras.Model): Trained autoencoder model.

    Returns:
        encoder (tf.keras.Model): Model mapping input -> latent vector.
    """

    encoder = tf.keras.Model(inputs = model.input,
                             outputs = model.get_layer('bottle_neck').output)
    
    latent_features = encoder.predict(X_reshaped, batch_size = batch_size)
    
    return encoder, latent_features


def init_cluster_centers(cfg:clusterConfig, latent_features):
    """
    Computes and initializes trainable cluster centers using K-Means.

    Args:
        n_clusters (int): Number of clusters to form.
        latent_features (np.ndarray or tf.Tensor): 
            Latent feature representations used as input to K-Means.

    Returns:
        tf.Variable: TensorFlow variable containing the initialized 
        cluster centers (shape: [n_clusters, feature_dim]).
        initial_labels (np.ndarray): Array of initial cluster labels for each input sample (shape: [num_samples]).
    """

    print("Running KMeans for clusters...")
    kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=20, random_state=42)
    initial_labels = kmeans.fit_predict(latent_features)
    m0 = kmeans.cluster_centers_.astype('float32')

    cluster_centers = tf.Variable(m0, trainable=True, name = "cluster_centers")

    return cluster_centers, initial_labels


def soft_assign(z, centers, alpha = 10):
    """
    Computes soft cluster assignments using the Student’s t-distribution.

    Used in IDEC (Improved Deep Embedded Clustering).

    Args:
        z (tf.Tensor): Latent representations (batch_size, latent_dim).
        cluster_centers (tf.Tensor): Cluster centroids (n_clusters, latent_dim).

    Returns:
        q (tf.Tensor): Soft assignment probabilities (batch_size, n_clusters).
    """
    z_exp = tf.expand_dims(z, 1)
    centers_exp = tf.expand_dims(centers, 0)  
    dist_sq = tf.reduce_sum(tf.square(z_exp - centers_exp), axis=2)
    q = 1.0 / (1.0 + dist_sq / alpha)
    q = q ** ((alpha + 1.0) / 2.0)
    q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    return q


def target_distribution(q):
    """
    Generates the target distribution 'p' for KL divergence in IDEC.

    Args:
        q (tf.Tensor): Soft cluster assignment (batch_size, n_clusters).

    Returns:
        p (tf.Tensor): Sharpened target distribution.
    """

    weight = tf.square(q) / tf.reduce_sum(q, axis=0)
    p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))
    return p
