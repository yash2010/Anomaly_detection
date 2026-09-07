import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
import mlflow
from src.model import soft_assign, target_distribution

def repulsion_loss(z, beta, eps = 1e-8):
    z_exp1 = tf.expand_dims(z, 1)
    z_exp2 = tf.expand_dims(z, 0)
    dist_sq = tf.reduce_sum(tf.square(z_exp1 - z_exp2), axis=2)
    mask = 1.0 - tf.eye(tf.shape(z)[0])
    dist_sq = tf.maximum(dist_sq, 1e-2)
    rep = mask / (dist_sq + eps)
    rep_loss = beta * tf.reduce_mean(rep)
    return rep_loss
    
def fine_tuning(X_reshaped, encoder, model, cluster_centers, optimizer, 
                batch_size, beta_rep, finetune_epochs, update_centers = False, 
                update_interval = None, momentum = 0.5, target_ratio = 0.1,
                **kwargs):
    """
    Encourages separation between latent vectors to prevent cluster overlap.
    Performs fine-tuning of the autoencoder using combined losses:
      - Reconstruction (MSE)
      - KL divergence (for clustering consistency)
      - Repulsion (for cluster separation)
      - Updates the clusters centers every x epoch (Optional)

    Args:
        X (np.ndarray): Training input data (samples, timesteps, 1).
        encoder (tf.keras.Model): Encoder part of autoencoder.
        model (tf.keras.Model): Full autoencoder.
        cluster_centers_var (tf.Variable): Trainable cluster centroids.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer (Adam).
        batch_size (int): Number of samples per training batch.
        beta_rep (float): Weight for repulsion loss.
        finetune_epochs (int): Number of fine-tuning epochs.
        update_centers (bool): Whether to update cluster centers dynamically.
        update_interval (int): Interval (epochs) to update cluster centers.

    Returns:
        tf.Tensor: Mean repulsion loss scalar.
        dict: Contains latent vectors, final labels, and loss logs.
    """
    @tf.function
    def _train_step(x):
        with tf.GradientTape() as tape:
                z = encoder(x, training=True)
                q = soft_assign(z, cluster_centers)
                p = tf.stop_gradient(target_distribution(q))
                kl_loss = tf.reduce_mean(tf.keras.losses.KLDivergence()(p, q))
                x_pred = model(x, training=True)
                recon_loss = tf.reduce_mean(tf.square(x - x_pred))
                rep_loss = repulsion_loss(z, beta_rep)
        
                alpha_kl = target_ratio * (recon_loss / (kl_loss + 1e-8))
                total_loss = recon_loss + alpha_kl * kl_loss + rep_loss

        grads = tape.gradient(total_loss, model.trainable_variables + [cluster_centers])
        optimizer.apply_gradients(zip(grads, model.trainable_variables + [cluster_centers]))
            
        return total_loss, kl_loss, recon_loss , rep_loss

    print("Phase 2: Fine-tuning clustering...")

    total_loss_log = []
    reconst_loss_log = []
    kl_loss_log = []
    repulsion_loss_log = []

    for e in range(finetune_epochs):
        epoch_total_loss = []
        epoch_recon_loss = []
        epoch_kl_loss = []
        epoch_rep_loss = []

        for i in range(0, len(X_reshaped), batch_size):
            batch_x = X_reshaped[i:i+batch_size]
            total_loss, kl_loss, reconst_loss, rep_loss = _train_step(batch_x)

            epoch_total_loss.append(total_loss.numpy())
            epoch_recon_loss.append(reconst_loss.numpy())
            epoch_kl_loss.append(kl_loss.numpy())
            epoch_rep_loss.append(rep_loss.numpy())
        
        # Calculate mean losses for the epoch
        mean_total = np.mean(epoch_total_loss)
        mean_recon = np.mean(epoch_recon_loss)
        mean_kl = np.mean(epoch_kl_loss)
        mean_rep = np.mean(epoch_rep_loss)

        total_loss_log.append(mean_total)
        reconst_loss_log.append(mean_recon)
        kl_loss_log.append(mean_kl)
        repulsion_loss_log.append(mean_rep)

        print(
        f"Epoch {e+1}/{finetune_epochs} - "
        f"Total Loss: {mean_total:.4f} | "
        f"Recon: {mean_recon:.4f} | "
        f"KL: {mean_kl:.6f} | "
        f"Repulsion: {mean_rep:.4f} | " )
        
        # MLflow logging
        mlflow.log_metrics({
            "total_loss": mean_total,
            "recon_loss": mean_recon,
            "kl_loss": mean_kl,
            "rep_loss": mean_rep
        }, step=e)


        # === OPTIONAL: UPDATE CLUSTER CENTERS ===
        if update_centers and update_interval and (e + 1) % update_interval == 0:
            
            print("Updating cluster centers with Kmeans...")
            latent_features = encoder.predict(X_reshaped, batch_size = batch_size)
            kmeans = KMeans(n_clusters=cluster_centers.shape[0], n_init=20, random_state=42)
            new_cluster_center = kmeans.fit(latent_features).cluster_centers_.astype('float32')
            updated_centers = momentum * cluster_centers.numpy() + (1 - momentum) * new_cluster_center
            cluster_centers.assign(updated_centers)
            print(f"Cluster centers updated at epoch {e + 1}")

    # === RETURN FINAL LATENT VECTORS AND CLUSTER LABELS ===
    final_latent = encoder.predict(X_reshaped, batch_size = batch_size, verbose = 0)
    final_labels = KMeans(n_clusters=cluster_centers.shape[0], n_init=20, random_state=42).fit_predict(final_latent)

    print("Clustering finished")

    return{"latent": final_latent,
           "labels": final_labels,
           "losses": {
               "total": total_loss_log,
               "recon": reconst_loss_log,
               "KL": kl_loss_log,
               "rep": repulsion_loss_log
           }
        }
