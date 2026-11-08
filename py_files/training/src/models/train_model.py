import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import LEARNING_RATE, NUM_EPOCHS, SCHEDULER_PATIENCE, EARLY_STOP_PATIENCE, THRESHOLD

def train_cnn_model(model, train_loader, val_loader, device, random_seed=42):
    """Train the SimpleCNN model on glass vial images."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = model.to(device)
    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    
    # Handle imbalanced dataset weighting
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE)

    history = {'best_val_f1_macro': 0.0, 'best_model_state': None}
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_running_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= THRESHOLD).float().view(-1)
            
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                logits = model(images)
                loss = criterion(logits, labels)
                val_running_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                predicted = (probs >= THRESHOLD).float().view(-1)

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')

        scheduler.step(epoch_val_f1_macro)

        if epoch_val_f1_macro > history['best_val_f1_macro']:
            history['best_val_f1_macro'] = epoch_val_f1_macro
            history['best_model_state'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

    if history['best_model_state'] is not None:
        model.load_state_dict(history['best_model_state'])

    return model, history

def tune_threshold(model, val_loader, device, threshold_range=None):
    """Tune the classification threshold to maximize macro F1."""
    if threshold_range is None:
        threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            logits = model(images)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)

    best_threshold, best_f1 = THRESHOLD, 0.0

    for t in threshold_range:
        preds = (all_probs >= t).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro')
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = t

    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_threshold_macro_f1", best_f1)
    
    return best_threshold

def evaluate_cnn_model(model, test_loader, device, threshold):
    """Evaluate the model on the test set."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(float).reshape(-1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'classification_report': classification_report(all_labels, all_preds, output_dict=True),
    }

def compute_baseline_statistics(model, data_loader, device, feature_layer="fc1", threshold=THRESHOLD):
    """Compute baseline statistics for drift detection."""
    model.eval()
    image_means, image_stds, confidences = [], [], []
    class_counts = {0: 0, 1: 0}
    prediction_counts = {0: 0, 1: 0}
    all_embeddings = []

    feature_extractor = getattr(model, feature_layer)
    embeddings = []
    
    def embedding_hook(module, input, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = feature_extractor.register_forward_hook(embedding_hook)

    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= threshold).astype(int)

                image_means.append(images.mean(dim=(0, 2, 3)).cpu().numpy())
                image_stds.append(images.std(dim=(0, 2, 3)).cpu().numpy())

                for label in labels.cpu().numpy().flatten():
                    class_counts[int(label)] += 1
                for pred in preds:
                    prediction_counts[int(pred)] += 1

                confidences.extend(probs)
                
                if embeddings:
                    all_embeddings.extend([e for e in embeddings])
                    embeddings.clear()

        image_means = np.array(image_means)
        image_stds = np.array(image_stds)

        baseline_stats = {
            'image_statistics': {
                'mean_r': float(np.mean(image_means[:, 0])),
                'mean_g': float(np.mean(image_means[:, 1])),
                'mean_b': float(np.mean(image_means[:, 2])),
                'std_r': float(np.mean(image_stds[:, 0])),
                'std_g': float(np.mean(image_stds[:, 1])),
                'std_b': float(np.mean(image_stds[:, 2])),
            },
            'class_distribution': {'class_0_count': class_counts[0], 'class_1_count': class_counts[1]},
            'prediction_distribution': {'pred_0_count': prediction_counts[0], 'pred_1_count': prediction_counts[1]},
            'confidence_statistics': {'mean': float(np.mean(confidences)), 'std': float(np.std(confidences))},
        }

        if all_embeddings:
            all_embeddings_array = np.concatenate(all_embeddings, axis=0)
            baseline_stats['embedding_statistics'] = {
                'mean': np.mean(all_embeddings_array, axis=0).tolist(),
                'std': np.std(all_embeddings_array, axis=0).tolist(),
                'shape': int(all_embeddings_array.shape[1]),
            }
            
        return baseline_stats
    finally:
        handle.remove()