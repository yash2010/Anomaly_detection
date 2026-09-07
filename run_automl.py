import os
import datetime
from collections import Counter
import numpy as np
import tensorflow as tf
import optuna
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from config import modelConfig, clusterConfig
from src.data_loader import load_fft_data, preprocessing
from src.model import autoencoder, get_latent_features, init_cluster_centers
from src.train import fine_tuning
from src.utils import save_path, plot_losses, plot_overall_reconstruction, Anomaly_detection_using_reconst_error, fine_tuned_latent, pretrained_latent, Reconstruction_error_distribution, anomalies_per_cluster, plotly_df, plot_signals
import random

# Set random seeds for reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

# Disable GPU if needed or configure it
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def objective(trial, X_reshaped):
    # ==== CONFIGURATION FROM OPTUNA ====
    tf.keras.backend.clear_session()
    bottleneck_dim = trial.suggest_categorical('bottleneck_dim', [16, 32, 64])
    lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2])
    beta_rep = trial.suggest_categorical('beta_rep', [0.001, 0.01, 0.1])
    n_clusters = trial.suggest_categorical('n_clusters', [3, 5, 7])
    
    # Fixed config
    # file_path = "fft_data.npy" # Ensure this file exists
    base_dir = "results_automl"
    min_samples = 10
    max_len = 1024
    max_freq = 2048
    batch_size = 128
    finetune_epochs = 5 # Reduced for speed in AutoML, increase for final run
    update_centers = True
    update_interval = 3

    # MLflow Run
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)

        # ==== STEP 1: Load and Preprocess Data ====
        # Data is now passed as an argument to the objective function
        # data = load_fft_data(file_path, min_samples)
        # X, y, Xscaled = preprocessing(data, max_len=max_len, max_freq=max_freq)
        # X_reshaped = np.expand_dims(Xscaled, axis=-1)

        # ==== STEP 2: Build and Train Autoencoder ====
        model_cfg = modelConfig(filters=[64, 32, 18, 8],
                                bottleneck_dim=bottleneck_dim,
                                lr=lr,
                                input_dim=X_reshaped.shape[1])
        model = autoencoder(model_cfg)

        print("Phase 1: Pretraining autoencoder...")
        history = model.fit(X_reshaped, X_reshaped,
                  epochs=10, # Reduced for AutoML
                  batch_size=batch_size,
                  validation_split=0.1,
                  shuffle=True,
                  verbose=0)
        
        pretrain_loss = history.history['loss'][-1]
        mlflow.log_metric("pretrain_loss", pretrain_loss)

        # ==== STEP 3: Extract Latent Features ====
        encoder, latent_features = get_latent_features(model, X_reshaped, batch_size=batch_size)

        # ==== STEP 4: Initialize Cluster Centers ====
        cluster_cfg = clusterConfig(n_clusters=n_clusters)
        cluster_centers_var, inital_labels = init_cluster_centers(cluster_cfg, latent_features)

        # ==== STEP 5: Fine-tuning with Clustering Loss ====
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        results = fine_tuning(X_reshaped, encoder, model, cluster_centers_var, optimizer,
                              batch_size=batch_size, beta_rep=beta_rep,
                              finetune_epochs=finetune_epochs,
                              update_centers=update_centers, update_interval=update_interval)

        final_loss = results["losses"]["total"][-1]
        score = silhouette_score(results["latent"], results["labels"], sample_size=2000, random_state=42)
        mlflow.log_metrics({"pretraining_loss": pretrain_loss,
                           "final_total_loss": final_loss,
                           "silhouette_score": score})
        
        # Save artifacts for the best trial or all trials
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_dir = os.path.join(base_dir, f"trial_{trial.number}_{timestamp}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Save model
        model.save(os.path.join(trial_dir, "model.h5"))
        mlflow.log_artifact(os.path.join(trial_dir, "model.h5"))

        return final_loss, -score

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results_automl", exist_ok=True)
    
    # MLflow Setup
    mlflow.set_experiment("Anomaly_Detection_AutoML")

    print("Loading data from inputX_compressed.npz and labelX.npy...")
    try:
        data_container = np.load('inputX_compressed.npz')
        X = data_container['X']
        y = np.load('labelX.npy')
        
        print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

        min_group_samples = 100
        group_counts = Counter(y.tolist())
        small_groups = {g: c for g, c in group_counts.items() if c < min_group_samples}
        if small_groups:
            print(f"Excluding groups with < {min_group_samples} samples: {small_groups}")
            keep_mask = np.array([group_counts[label] >= min_group_samples for label in y])
            X, y = X[keep_mask], y[keep_mask]
            print(f"Remaining after exclusion: X shape: {X.shape}, y shape: {y.shape}")

        # Scale the data
        print("Scaling data...")
        scaler = StandardScaler()
        Xscaled = scaler.fit_transform(X)
        
        # Reshape for Autoencoder (samples, timesteps, features)
        X_reshaped = np.expand_dims(Xscaled, axis=-1)
        print(f"Data preprocessed. X_reshaped shape: {X_reshaped.shape}")

    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Optuna Setup
    study = optuna.create_study(directions=["minimize", "minimize"], storage="sqlite:///db.sqlite3", study_name="anomaly_detection_study", load_if_exists=True)

    study.optimize(lambda trial: objective(trial, X_reshaped), n_trials=5) # Run 5 trials for demonstration

    print("Best trials:")
    for trial in study.best_trials:
        print("  Values (total_loss, -silhouette_score): ", trial.values)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
