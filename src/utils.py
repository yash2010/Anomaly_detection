import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import umap
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv1D

def save_path(model, encoder, base_dir, bottleneck_dim, suffix):
    """
    Saves a visualization of the autoencoder model architecture and 
    prepares a structured directory for storing all related outputs.

    Args:
        model (tf.keras.Model): The full autoencoder model.
        encoder (tf.keras.Model): The encoder portion of the autoencoder.
        base_dir (str): Base directory where the outputs will be saved.
        bottleneck_dim (int): Size of the latent/bottleneck layer.
        suffix (str): A suffix for naming the folder (e.g., timestamp or experiment name).

    Returns:
        image_dir (str): Path to the folder where the model image is saved.
    """

    encoder_filters = []
    for layer in encoder.layers:
        if isinstance(layer, Conv1D):
            encoder_filters.append(layer.filters)
    
    arch_str = "_".join(map(str, encoder_filters))
    image_dir = os.path.join(base_dir, f"{arch_str}_D{bottleneck_dim}_with_{suffix} ")
    os.makedirs(image_dir, exist_ok=True)

    image_file = os.path.join(image_dir, "model.png")
    try:
        plot_model(
            model,
            to_file = image_file,
            show_shapes = True,
            show_layer_names = True,
            rankdir = "LR"
        )
        print(f"Model saved at: {image_file}")
    except Exception as e:
        print(f"Could not save model plot (graphviz might be missing): {e}")

    return image_dir


def plot_losses(total_loss, recon_loss, kl_loss, rep_loss, image_dir):
    """
    Plots total, reconstruction, KL, and repulsion losses per epoch and saves the figure.

    Args:
        total_loss (list or np.ndarray): List of total loss values per epoch.
        recon_loss (list or np.ndarray): List of reconstruction loss values per epoch.
        kl_loss (list or np.ndarray): List of KL divergence loss values per epoch.
        rep_loss (list or np.ndarray): List of repulsion loss values per epoch.
        image_dir (str): Directory path where the plot image will be saved.
    """

    plt.figure(figsize=(10,6))
    plt.plot(total_loss, label='Total Loss')
    plt.plot(recon_loss, label='Reconstruction Loss')
    plt.plot(kl_loss, label='KL Loss')
    plt.plot(rep_loss, label='Repulsion Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Fine-tuning Loss Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()


def plot_overall_reconstruction(model, X_reshaped, image_dir, sample_idx = None, dpi = 300):
    """
    Visualizes and saves one sample reconstruction comparison
    (original vs reconstructed signal).

    Args:
        X (np.ndarray): Input test data (samples, timesteps, 1).
        model (tf.keras.Model): Trained autoencoder model.
        image_dir (str): Directory to save output images.
    
    Returns:
        pred (np.ndarray): Reconstructed signals for all input samples, shape same as X_reshaped.
    """

    pred = model.predict(X_reshaped)

    if sample_idx is None:
        sample_idx = np.random.randint(len(X_reshaped))

    plt.figure()
    plt.plot(X_reshaped[sample_idx].squeeze(), label='Original', color='blue')
    plt.plot(pred[sample_idx].squeeze(), label='Reconstructed', color='red', linestyle='--')
    plt.title("Original vs Reconstructed")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()

    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()

    
    print(f"{title_str} saved at: {file_path}")
    
    return pred 


def Anomaly_detection_using_reconst_error(X_reshaped, pred, image_dir):
    """
    Detects anomalies based on reconstruction error from the autoencoder and visualizes the results.

    Args:
        X_reshaped (np.ndarray): Original input data, shape (num_samples, sequence_length, 1).
        pred (np.ndarray): Reconstructed data from the autoencoder, same shape as X_reshaped.
        image_dir (str): Directory path to save the plot of reconstruction errors.

    Returns:
        reconst_error (np.ndarray): Reconstruction error (MSE) for each sample.
        anomalies (np.ndarray): Boolean array indicating which samples are considered anomalies
                                (True if the reconstruction error exceeds the 99th percentile threshold).
    """

    reconst_error = np.mean(np.square(X_reshaped - pred), axis=1).squeeze()
    threshold = np.percentile(reconst_error, 99)
    anomalies = reconst_error > threshold

    colors = ['red' if i else 'blue' for i in anomalies]
    plt.figure(figsize=(12,6))
    plt.scatter(range(len(reconst_error)), reconst_error, c=colors, label='Data Points')
    plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    plt.title('Anomaly Detection using Reconstruction Error')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.xlim(left=0)

    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()

    print(f"{title_str} saved at: {file_path}")
    return reconst_error, anomalies


def fine_tuned_latent(final_latent, image_dir, final_labels):
    """
    Reduces latent features to 2D using UMAP for visualization.

    Args:
        latent (np.ndarray): Latent vectors.
        labels (np.ndarray): Cluster labels.
        image_dir (str): Path to save the plot
    Returns:
        reduced (np.ndarray): 2D UMAP embedding of the latent features, shape (num_samples, 2) 
    """

    reduced = umap.UMAP(n_components = 2, random_state = 42).fit_transform(final_latent)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=final_labels, cmap='tab10', s=5)
    plt.title('Clustering on Latent Space Finetuned')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()

    print(f"{title_str} saved at: {file_path}")
    return reduced


def pretrained_latent(latent_features, initial_labels, image_dir):
    """
    Reduces pretrained latent features to 2D using UMAP for visualization.

    Args:
        latent_features (np.ndarray): Latent vectors from the pretrained autoencoder.
        initial_labels (np.ndarray): Cluster or initial labels for coloring points.
        image_dir (str): Directory path to save the plot.

    Returns:
        reduced_pretrained (np.ndarray): 2D UMAP embedding of the latent features, shape (num_samples, 2).
    """

    reduced_pretrained = umap.UMAP(n_components=2, random_state=42).fit_transform(latent_features)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_pretrained[:, 0], reduced_pretrained[:, 1], s=5, c=initial_labels, cmap='tab10')
    plt.title("Clustering on Latent Space (Pretrained)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()

    print(f"{title_str} saved at: {file_path}")
    return reduced_pretrained


def Reconstruction_error_distribution(final_labels, reconst_error, image_dir):
    """
    Plots reconstruction error distribution for each cluster.

    Args:
        final_labels (np.ndarray): Cluster labels for each sample.
        reconst_error (np.ndarray): Reconstruction errors of all samples.
        image_dir (str): Directory path to save the histogram plot. 
    
    """
    
    plt.figure()
    thresh = {}
    for cluster_id in np.unique(final_labels):
        cluster_error = reconst_error[final_labels == cluster_id]
        cluster_thresh = np.percentile(cluster_error, 99)  
        thresh[cluster_id] = cluster_thresh  
        plt.hist(cluster_error, bins=50, alpha=0.6, label=f"Cluster {cluster_id}")
        
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution per Cluster")
    plt.legend()

    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()

    print(f"{title_str} saved at: {file_path}")


def anomalies_per_cluster(final_labels, reconst_error, image_dir):
    """
    Detects anomalies cluster-wise using reconstruction error and visualizes them.

    Args:
        final_labels (np.ndarray): Cluster labels for each sample.
        reconst_error (np.ndarray): Reconstruction errors of all samples.
        image_dir (str): Directory path to save the plot.

    Returns:
        cluster_anomalies (np.ndarray): Boolean array marking anomalies per cluster.
    """

    unique_clusters = np.unique(final_labels)
    cluster_thresholds = {}
    cluster_anomalies = np.zeros_like(reconst_error, dtype=bool)
    for cluster_id in unique_clusters:
        cluster_mask = (final_labels == cluster_id)
        cluster_error = reconst_error[cluster_mask]

        threshold = np.percentile(cluster_error, 99)
        cluster_thresholds[cluster_id] = threshold

        cluster_anomalies[cluster_mask] = cluster_error > threshold

    plt.figure(figsize=(14, 6))
    colors = plt.get_cmap('tab10')

    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = (final_labels == cluster_id)
        normal_mask = cluster_mask & (~cluster_anomalies)

        cluster_indices = np.where(normal_mask)[0]
        cluster_errors = reconst_error[normal_mask]

        plt.scatter(cluster_indices, cluster_errors,
                    color=colors(i), label=f'Cluster {cluster_id}', alpha=0.6)

    anomaly_indices = np.where(cluster_anomalies)[0]
    plt.scatter(anomaly_indices, reconst_error[anomaly_indices],
                facecolors='black', edgecolors='red', s=80, label='Anomalies')
    

    plt.title('Cluster-wise Reconstruction Errors with Anomalies')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()

    print(f"{title_str} saved at: {file_path}")
    return cluster_anomalies


def plotly_df(data, reduced, fft_labels, final_labels, reconst_error, anomalies, image_dir):
    """
    Generates interactive 2D scatter plots of the latent space using Plotly with multiple metadata options.

    Args:
        data (dict): Original FFT dataset containing metadata.
        reduced (np.ndarray): 2D UMAP latent embeddings (num_samples x 2).
        fft_labels (list or np.ndarray): Original FFT labels for each sample.
        final_labels (np.ndarray): Cluster labels after fine-tuning.
        reconst_error (np.ndarray): Reconstruction errors for all samples.
        anomalies (np.ndarray): Boolean array indicating anomalies per sample.
        image_dir (str): Directory to save Plotly images (.png) and interactive HTML files.

    Returns:
        latent_df (pd.DataFrame): DataFrame containing 2D latent coordinates and metadata for each sample.
        unique_ids (list): List of unique IDs parsed from the data labels.
        kugel (list): Metadata corresponding to 'Kugel' for each sample.
        steigung (list): Metadata corresponding to 'steigung' for each sample.
    """

    unique_ids = []
    kugel = []
    steigung = []

    for folder, data_list in data.items():
        for label, (freq, fft_mag) in data_list:
            match = re.search(r'\\(\d+)_', label)
            unique_ids.append(match.group(1) if match else None)

    for i in unique_ids:
        if i in ['7806052964' , '7806052788' , '7806052787' , '7806052965' ]:
            steigung.append('8000')
        elif i in ['7806052540' , '7806052541']:
            steigung.append('9000')
        else:
             steigung.append('Unknown')


    for i in unique_ids:
        if i in ['7806052965' , '7806052788' , '7806052787', '7806052964']:
            kugel.append('4762')
        elif i in ['7806052540' , '7806052541']:
            kugel.append('3969')
        else:
            kugel.append('Unknown')

    latent_df = pd.DataFrame({
    'PC1': reduced[:,0],
    'PC2': reduced[:,1],
    'U_ID': unique_ids,
    'Label': fft_labels,
    'Index': range(len(fft_labels)),
    'Cluster': final_labels,
    'Kugel': kugel,
    'steigung': steigung,
    'Reconstruction Error': reconst_error,
    'Is anomaly': anomalies})

    color_list = ['Cluster', 'Label', 'U_ID','steigung', 'Kugel', 'Is anomaly'] 
    for color in color_list:
        if color in ['Cluster', 'Label']:  
            latent_df[color] = latent_df[color].astype(str)

        fig = px.scatter(
            latent_df, 
            x='PC1', 
            y='PC2',
            color=color,
            hover_data=['Label', 'Index', 'Reconstruction Error', 'U_ID','steigung', 'Kugel', 'Is anomaly'],  
            title="Latent Space Clustering (with FFT Metadata)"
        )
        fig.update_layout(legend_title_text=color)
        # fig.show()
        filename_image = f"{color}_plotly.png"
        filename_html = f"{color}_plotly.html"
        full_path_image = os.path.join(image_dir, filename_image)
        full_path_html = os.path.join(image_dir, filename_html)
        # fig.write_image(full_path_image) # Requires kaleido
        fig.write_html(full_path_html)

    return latent_df, unique_ids, kugel, steigung


def plot_signals(final_labels, X, cluster_anomalies, image_dir):
    """
    Plots representative signals for each cluster, including mean normal, normal, anomaly, 
    and differences from the mean.

    Args:
        final_labels (np.ndarray): Cluster labels for each sample.
        X (np.ndarray): Original signals (shape: num_samples x signal_length).
        cluster_anomalies (np.ndarray): Boolean array indicating anomalies per sample.
        image_dir (str): Directory path to save the plots.
    """
    
    n_clusters = len(np.unique(final_labels))
    fig, axes = plt.subplots(n_clusters, 5, figsize=(50, 4 * n_clusters), sharex=True)

    if n_clusters == 1:  
        axes = np.expand_dims(axes, axis=0)

    for i, cluster_id in enumerate(np.unique(final_labels)):
        cluster_mask = (final_labels == cluster_id)
        normal_mask = cluster_mask & (~cluster_anomalies)
        anomaly_mask = cluster_mask & cluster_anomalies

        if np.any(normal_mask):
            mean_signal = X[normal_mask].mean(axis=0)
            axes[i, 0].plot(mean_signal, color='blue')
            axes[i, 0].set_title(f"Cluster {cluster_id} - Mean Normal")
            normal = X[normal_mask][0]
            axes[i, 1].plot(normal, color='green')
            axes[i, 1].set_title(f"Cluster {cluster_id} - Normal")
        else:
            mean_signal = np.zeros(X.shape[1])
            normal = np.zeros(X.shape[1])
            axes[i, 0].set_title(f"Cluster {cluster_id} - No Normals")
            axes[i, 1].set_title(f"Cluster {cluster_id} - No Normals")

        if np.any(anomaly_mask):
            anomaly = X[anomaly_mask][0] # Take first anomaly
            axes[i, 2].plot(anomaly, color='red')
            axes[i, 2].set_title(f"Cluster {cluster_id} - Anomaly")
        else:
            anomaly = np.zeros(X.shape[1])
            axes[i, 2].set_title(f"Cluster {cluster_id} - No Anomalies")
        
        axes[i, 3].plot(normal, color="green")
        axes[i, 3].plot(mean_signal, color='black')
        axes[i, 3].set_title(f"Cluster {cluster_id} - Normal Diff (Mean - Normal)")
        axes[i, 3].grid(True)

        axes[i, 4].plot(anomaly, color="red")
        axes[i, 4].plot(mean_signal, color='black')
        axes[i, 4].set_title(f"Cluster {cluster_id} - Anomaly Diff (Mean - Anomaly)")
        axes[i, 4].grid(True)

        for j in range(5):
            axes[i, j].grid(True)
            axes[i, j].set_xlim(0, X.shape[1])
            axes[i, j].set_ylim(0, 55)

    plt.tight_layout()
    
    title_str = plt.gca().get_title().replace(" ", "_")
    file_path = os.path.join(image_dir, f"{title_str}.png")
    plt.savefig(file_path, dpi=300)
    # plt.show()
