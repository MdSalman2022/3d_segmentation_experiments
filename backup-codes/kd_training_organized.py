"""
Knowledge Distillation Training Script
Organized version with centralized output management
All outputs saved to ./outputs directory
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    f1_score, cohen_kappa_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, brier_score_loss,
    balanced_accuracy_score, top_k_accuracy_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_convnext
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Concatenate, Layer,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.losses import KLDivergence, SparseCategoricalCrossentropy
import optuna
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset Configuration
DATASET_DIR = "./leukemia"
CLASS_NAMES = ["Benign", "Pre", "Pro", "Early"]
INPUT_SHAPE = (224, 224, 3)

# Training Configuration
BATCH_SIZE = 16
EPOCHS = 10
FINAL_EPOCHS = 10
LEARNING_RATE = 1e-5

# Distillation Configuration
DISTILLATION_TEMPERATURE = 5.0
DISTILLATION_ALPHA = 0.9

# Optimized Hyperparameters (from Optuna)
OPTIMIZED_LR = 3.9342870981686775e-05
OPTIMIZED_LORA_RANK = 8
OPTIMIZED_LORA_ALPHA = 48
OPTIMIZED_ATTENTION_HEADS = 8
OPTIMIZED_WINDOW_SIZE = 3
OPTIMIZED_GLOBAL_TOKENS = 16
OPTIMIZED_DROPOUT = 0.2878660419841268
OPTIMIZED_DENSE_UNITS = 128
OPTIMIZED_DENSE_DROPOUT = 0.447675960241552

# Student Model Optimized Hyperparameters
OPTIMIZED_STUDENT_DENSE = 512
OPTIMIZED_STUDENT_DROPOUT = 0.6646
OPTIMIZED_STUDENT_LR = 0.000076
OPTIMIZED_STUDENT_LORA_RANK = 8
OPTIMIZED_STUDENT_LORA_ALPHA = 48
OPTIMIZED_TEMPERATURE = 7.80
OPTIMIZED_DISTILL_ALPHA = 0.782

# Output Configuration
OUTPUT_DIR = "./outputs"
TEACHER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "teacher")
STUDENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "student")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories
for dir_path in [OUTPUT_DIR, TEACHER_OUTPUT_DIR, STUDENT_OUTPUT_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Set plot parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

print(f"Output directories created:")
print(f"  - {OUTPUT_DIR}")
print(f"  - {TEACHER_OUTPUT_DIR}")
print(f"  - {STUDENT_OUTPUT_DIR}")
print(f"  - {MODELS_DIR}")
print(f"  - {PLOTS_DIR}")

# =============================================================================
# GPU CONFIGURATION
# =============================================================================

def configure_gpu():
    """Configure GPU memory growth"""
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("Available GPUs:", physical_devices)
    print(f"Number of GPUs detected: {len(physical_devices)}" if physical_devices else "No GPUs detected, falling back to CPU.")
    return physical_devices

# =============================================================================
# CUSTOM LAYERS
# =============================================================================

class LoRADense(Layer):
    """LoRA-enhanced Dense Layer"""
    def __init__(self, units, rank=16, alpha=32, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.activation = activation
        self.scaling = alpha / rank
        self.base_dense = Dense(units, activation=activation)
    
    def build(self, input_shape):
        self.lora_A = self.add_weight(
            name='lora_A',
            shape=(input_shape[-1], self.rank),
            initializer='random_normal',
            trainable=True
        )
        self.lora_B = self.add_weight(
            name='lora_B',
            shape=(self.rank, self.units),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        base_output = self.base_dense(inputs)
        lora_output = tf.matmul(inputs, tf.matmul(self.lora_A, self.lora_B)) * self.scaling
        if self.activation == 'relu':
            lora_output = tf.nn.relu(lora_output)
        return base_output + lora_output


class LoRAFocalSelfAttention(Layer):
    """Enhanced Focal Self-Attention with LoRA"""
    def __init__(self, embed_dim, num_heads=8, window_size=3, num_global_tokens=16,
                 dropout_rate=0.1, lora_rank=16, lora_alpha=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_global = num_global_tokens
        self.dropout_rate = dropout_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        self.local_mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
        )
        self.global_mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
        )
        
        self.base_proj = Dense(embed_dim, activation=None, name='base_projection')
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)
    
    def build(self, input_shape):
        _, H, W, C = input_shape
        self.H, self.W = H, W
        
        self.global_tokens = self.add_weight(
            shape=(1, self.num_global, C),
            initializer='random_normal',
            trainable=True,
            name='global_tokens'
        )
        
        self.lora_A = self.add_weight(
            name='lora_proj_A',
            shape=(self.embed_dim, self.lora_rank),
            initializer='random_normal',
            trainable=True
        )
        self.lora_B = self.add_weight(
            name='lora_proj_B',
            shape=(self.lora_rank, self.embed_dim),
            initializer='zeros',
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, x, training=False):
        B = tf.shape(x)[0]
        
        # Local patches
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.window_size, self.window_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        q_local = tf.reshape(x, [B, self.H * self.W, self.embed_dim])
        kv_local = tf.reshape(patches, [B, self.H * self.W * self.window_size**2, self.embed_dim])
        out_local = self.local_mha(query=q_local, key=kv_local, value=kv_local, training=training)
        
        # Global tokens cross-attention
        global_tokens = tf.tile(self.global_tokens, [B, 1, 1])
        out_global = self.global_mha(query=q_local, key=global_tokens, value=global_tokens, training=training)
        
        # Combine attention outputs
        combined_attention = out_local + out_global
        
        # LoRA-enhanced projection
        base_output = self.base_proj(combined_attention)
        lora_scaling = self.lora_alpha / self.lora_rank
        lora_output = tf.matmul(combined_attention, tf.matmul(self.lora_A, self.lora_B)) * lora_scaling
        enhanced_output = base_output + lora_output
        
        # Apply dropout, reshape, and normalize
        enhanced_output = self.dropout(enhanced_output, training=training)
        enhanced_output = tf.reshape(enhanced_output, [B, self.H, self.W, self.embed_dim])
        
        return self.norm(x + enhanced_output)


class LoRAStudentDense(Layer):
    """LoRA-Enhanced Dense Layer for Student Model"""
    def __init__(self, units, rank=8, alpha=16, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.activation = activation
        self.scaling = alpha / rank
        self.base_dense = Dense(units, activation=activation)
    
    def build(self, input_shape):
        self.lora_A = self.add_weight(
            name='lora_A',
            shape=(input_shape[-1], self.rank),
            initializer='random_normal',
            trainable=True
        )
        self.lora_B = self.add_weight(
            name='lora_B',
            shape=(self.rank, self.units),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        base_output = self.base_dense(inputs)
        lora_output = tf.matmul(inputs, tf.matmul(self.lora_A, self.lora_B)) * self.scaling
        if self.activation == 'relu':
            lora_output = tf.nn.relu(lora_output)
        return base_output + lora_output

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_dataset():
    """Load and split dataset"""
    image_paths = []
    labels = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(class_name)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return X_train_paths, X_test_paths, y_train, y_test, label_encoder


def load_and_preprocess_teacher_train(path, label):
    """Preprocessing for teacher model training (WITH augmentation)"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img_eff = preprocess_efficientnet(img)
    img_conv = preprocess_convnext(img)
    return (img_eff, img_conv), label


def load_and_preprocess_student_train(path, label):
    """Preprocessing for student model training (WITH augmentation)"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img_student = preprocess_efficientnet(img)
    return img_student, label


def load_and_preprocess_teacher_test(path, label):
    """Preprocessing for teacher model testing (WITHOUT augmentation)"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
    img_eff = preprocess_efficientnet(img)
    img_conv = preprocess_convnext(img)
    return (img_eff, img_conv), label


def load_and_preprocess_student_test(path, label):
    """Preprocessing for student model testing (WITHOUT augmentation)"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
    img_student = preprocess_efficientnet(img)
    return img_student, label


def create_datasets(X_train_paths, X_test_paths, y_train, y_test):
    """Create TensorFlow datasets"""
    # Training datasets (WITH augmentation)
    train_dataset_teacher = (
        tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
        .map(load_and_preprocess_teacher_train, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(buffer_size=len(X_train_paths))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    train_dataset_student = (
        tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
        .map(load_and_preprocess_student_train, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(buffer_size=len(X_train_paths))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Test datasets (WITHOUT augmentation)
    test_dataset_teacher = (
        tf.data.Dataset.from_tensor_slices((X_test_paths, y_test))
        .map(load_and_preprocess_teacher_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    test_dataset_student = (
        tf.data.Dataset.from_tensor_slices((X_test_paths, y_test))
        .map(load_and_preprocess_student_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return train_dataset_teacher, train_dataset_student, test_dataset_teacher, test_dataset_student

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def create_lora_enhanced_teacher():
    """Create LoRA-enhanced teacher model"""
    base_model_eff = tf.keras.applications.EfficientNetB4(
        weights='imagenet', include_top=False, input_shape=INPUT_SHAPE
    )
    base_model_conv = tf.keras.applications.ConvNeXtBase(
        weights='imagenet', include_top=False, input_shape=INPUT_SHAPE
    )
    
    # Apply LoRA-enhanced Focal Self-Attention
    eff_attention = LoRAFocalSelfAttention(
        embed_dim=base_model_eff.output_shape[-1],
        num_heads=OPTIMIZED_ATTENTION_HEADS,
        window_size=OPTIMIZED_WINDOW_SIZE,
        num_global_tokens=OPTIMIZED_GLOBAL_TOKENS,
        dropout_rate=OPTIMIZED_DROPOUT,
        lora_rank=OPTIMIZED_LORA_RANK,
        lora_alpha=OPTIMIZED_LORA_ALPHA
    )(base_model_eff.output)
    
    conv_attention = LoRAFocalSelfAttention(
        embed_dim=base_model_conv.output_shape[-1],
        num_heads=OPTIMIZED_ATTENTION_HEADS,
        window_size=OPTIMIZED_WINDOW_SIZE,
        num_global_tokens=OPTIMIZED_GLOBAL_TOKENS,
        dropout_rate=OPTIMIZED_DROPOUT,
        lora_rank=OPTIMIZED_LORA_RANK,
        lora_alpha=OPTIMIZED_LORA_ALPHA
    )(base_model_conv.output)
    
    # Pooling and fusion
    x_eff = GlobalAveragePooling2D()(eff_attention)
    x_conv = GlobalAveragePooling2D()(conv_attention)
    concatenated = Concatenate()([x_eff, x_conv])
    
    # LoRA-enhanced classification head
    x = LoRADense(
        units=OPTIMIZED_DENSE_UNITS,
        rank=OPTIMIZED_LORA_RANK,
        alpha=OPTIMIZED_LORA_ALPHA,
        activation='relu'
    )(concatenated)
    x = Dropout(OPTIMIZED_DENSE_DROPOUT)(x)
    output = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    return Model(inputs=[base_model_eff.input, base_model_conv.input], outputs=output)


def create_student_model():
    """Create student model (EfficientNetB1)"""
    base_model_student = tf.keras.applications.EfficientNetB1(
        weights='imagenet', include_top=False, input_shape=INPUT_SHAPE
    )
    
    x = GlobalAveragePooling2D()(base_model_student.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    return Model(inputs=base_model_student.input, outputs=output)

# =============================================================================
# TRAINING CALLBACKS
# =============================================================================

class ProgressLogger(Callback):
    """Progress tracking callback"""
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting Epoch {epoch + 1}/{self.params['epochs']}")
        device = tf.test.gpu_device_name() if tf.config.list_physical_devices('GPU') else 'CPU'
        print(f"Training on: {device}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nFinished Epoch {epoch + 1}/{self.params['epochs']}")
        print(f" - Loss: {logs.get('loss', 0):.4f}")
        print(f" - Accuracy: {logs.get('accuracy', 0):.4f}")
        print(f" - Val Loss: {logs.get('val_loss', 0):.4f}")
        print(f" - Val Accuracy: {logs.get('val_accuracy', 0):.4f}")

# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def save_confusion_matrix(y_true, y_pred, save_path, title):
    """Save confusion matrix plot"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()


def save_roc_curves(y_true_bin, y_prob, save_path, title):
    """Save ROC curves"""
    n_classes = len(CLASS_NAMES)
    fpr, tpr, roc_auc = {}, {}, {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    
    plt.figure(figsize=(12, 10))
    plt.plot(fpr['micro'], tpr['micro'],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    return roc_auc


def save_precision_recall_curves(y_true_bin, y_prob, save_path, title):
    """Save Precision-Recall curves"""
    n_classes = len(CLASS_NAMES)
    precision, recall, average_precision = {}, {}, {}
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
    
    precision['micro'], recall['micro'], _ = precision_recall_curve(
        y_true_bin.ravel(), y_prob.ravel()
    )
    average_precision['micro'] = average_precision_score(y_true_bin, y_prob, average='micro')
    
    plt.figure(figsize=(12, 10))
    plt.step(recall['micro'], precision['micro'], where='post',
             label=f'Micro-average PR (AP = {average_precision["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.step(recall[i], precision[i], color=color, alpha=0.3, where='post')
        plt.fill_between(recall[i], precision[i], step='post', alpha=0.2, color=color)
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f"{CLASS_NAMES[i]} (AP = {average_precision[i]:.3f})")
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    return average_precision


def save_training_curves(history_dict, save_dir, model_name):
    """Save training accuracy and loss curves"""
    epochs = range(1, len(history_dict['accuracy']) + 1)
    
    # Accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history_dict['accuracy'], 'bo-', label='Training Accuracy',
             linewidth=2, markersize=6)
    plt.plot(epochs, history_dict['val_accuracy'], 'ro-', label='Validation Accuracy',
             linewidth=2, markersize=6)
    plt.title(f'{model_name} Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_accuracy.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/training_accuracy.pdf", bbox_inches='tight')
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss',
             linewidth=2, markersize=6)
    plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss',
             linewidth=2, markersize=6)
    plt.title(f'{model_name} Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_loss.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/training_loss.pdf", bbox_inches='tight')
    plt.close()


def comprehensive_evaluation(model, test_dataset, model_name, output_dir):
    """Perform comprehensive model evaluation and save all results"""
    print(f"\n{'='*60}")
    print(f"Comprehensive Evaluation: {model_name}")
    print(f"{'='*60}")
    
    # Evaluate model
    print("Evaluating model on test dataset...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    print(f"{model_name} Test Loss: {test_loss:.4f}")
    
    # Generate predictions
    y_true, y_pred, y_prob = [], [], []
    for x, y in test_dataset:
        y_true.extend(y.numpy())
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    top3_acc = top_k_accuracy_score(y_true, y_prob, k=3)
    
    print(f"\nBasic Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Weighted): {recall_weighted:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    # Save confusion matrix
    save_confusion_matrix(y_true, y_pred,
                         f"{output_dir}/01_confusion_matrix",
                         f"{model_name} Confusion Matrix")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("\nClassification Report:")
    print(report)
    
    with open(f"{output_dir}/classification_report.txt", 'w') as f:
        f.write(f"{model_name} Classification Report\n")
        f.write("="*50 + "\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1:.4f}\n")
        f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
        f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n\n")
        f.write(report)
    
    # ROC curves
    n_classes = len(CLASS_NAMES)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    roc_auc = save_roc_curves(y_true_bin, y_prob,
                              f"{output_dir}/02_roc_curves",
                              f"{model_name} ROC Curves (One-vs-Rest)")
    
    # Precision-Recall curves
    average_precision = save_precision_recall_curves(y_true_bin, y_prob,
                                                     f"{output_dir}/03_precision_recall_curves",
                                                     f"{model_name} Precision-Recall Curves")
    
    # Calibration plot
    y_prob_max = np.max(y_prob, axis=1)
    prob_true, prob_pred = calibration_curve(y_true == y_pred, y_prob_max, n_bins=10, normalize=True)
    
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='gray', label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'{model_name} Calibration Plot', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_calibration_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/04_calibration_plot.pdf", bbox_inches='tight')
    plt.close()
    
    # Confidence histogram
    plt.figure(figsize=(12, 8))
    plt.hist(y_prob_max, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(y_prob_max), color='red', linestyle='--', linewidth=2,
                label=f'Mean Confidence: {np.mean(y_prob_max):.3f}')
    plt.xlabel('Prediction Confidence (Max Probability)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name} Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_confidence_histogram.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/05_confidence_histogram.pdf", bbox_inches='tight')
    plt.close()
    
    # t-SNE visualization
    print("Extracting embeddings for t-SNE...")
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    
    features, labels_tsne = [], []
    for x, y in test_dataset:
        feat = feature_extractor.predict(x, verbose=0)
        features.extend(feat)
        labels_tsne.extend(y.numpy())
    
    features = np.array(features)
    labels_tsne = np.array(labels_tsne)
    
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_tsne,
                         cmap='tab10', alpha=0.7, s=20)
    cbar = plt.colorbar(scatter, ticks=range(n_classes), label='Class')
    cbar.ax.set_yticklabels(CLASS_NAMES)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f't-SNE Visualization of {model_name} Embeddings', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_tsne_embeddings.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/06_tsne_embeddings.pdf", bbox_inches='tight')
    plt.close()
    
    # Comprehensive metrics table
    brier_scores = [brier_score_loss(y_true_bin[:, i], y_prob[:, i]) for i in range(n_classes)]
    
    error_rates = {}
    for i in range(n_classes):
        class_mask = y_true == i
        class_errors = np.sum(y_pred[class_mask] != i) / np.sum(class_mask) if np.sum(class_mask) > 0 else 0
        error_rates[CLASS_NAMES[i]] = class_errors
    
    metrics_data = {
        'Class': CLASS_NAMES,
        'ROC AUC': [roc_auc[i] for i in range(n_classes)],
        'Average Precision': [average_precision[i] for i in range(n_classes)],
        'Brier Score': brier_scores,
        'Error Rate': [error_rates[CLASS_NAMES[i]] for i in range(n_classes)]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.loc['Micro-average'] = [
        'Micro-average',
        roc_auc['micro'],
        average_precision['micro'],
        np.mean(brier_scores),
        np.mean(list(error_rates.values()))
    ]
    
    print("\nComprehensive Metrics Summary:")
    print(metrics_df.to_string(index=False))
    
    metrics_df.to_csv(f"{output_dir}/comprehensive_metrics.csv", index=False)
    
    with open(f"{output_dir}/comprehensive_metrics.txt", 'w') as f:
        f.write(f"{model_name} Comprehensive Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Overall Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Overall Test Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1:.4f}\n")
        f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
        f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n")
        f.write(f"Mean Prediction Confidence: {np.mean(y_prob_max):.4f}\n")
        f.write(f"Average Brier Score: {np.mean(brier_scores):.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-"*40 + "\n")
        f.write(metrics_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete! Results saved to {output_dir}")
    print(f"{'='*60}")
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'metrics': {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision_weighted,
            'recall': recall_weighted,
            'balanced_accuracy': balanced_acc,
            'top3_accuracy': top3_acc
        }
    }

# =============================================================================
# DISTILLATION UTILITIES
# =============================================================================

def distillation_loss(y_true, y_pred, teacher_logits, temperature=OPTIMIZED_TEMPERATURE, alpha=OPTIMIZED_DISTILL_ALPHA):
    """Custom distillation loss"""
    y_true = tf.cast(y_true, tf.int32)
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(y_pred / temperature, axis=-1)
    soft_loss = KLDivergence()(teacher_probs, student_probs) * (temperature ** 2)
    hard_loss = SparseCategoricalCrossentropy()(y_true, y_pred)
    return alpha * soft_loss + (1 - alpha) * hard_loss


@tf.function
def train_step(student_model, teacher_model, optimizer, x, y):
    """Custom training step for knowledge distillation"""
    teacher_inputs = x
    student_inputs = preprocess_efficientnet(x[0])
    
    with tf.GradientTape() as tape:
        teacher_logits = teacher_model(teacher_inputs, training=False)
        student_logits = student_model(student_inputs, training=True)
        total_loss = distillation_loss(y, student_logits, teacher_logits)
    
    gradients = tape.gradient(total_loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
    
    predictions = tf.argmax(student_logits, axis=1, output_type=tf.int32)
    y = tf.cast(y, tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))
    
    return total_loss, accuracy

# =============================================================================
# MAIN TRAINING FUNCTIONS
# =============================================================================

def train_teacher_model(train_dataset, test_dataset):
    """Train teacher model"""
    print("\n" + "="*60)
    print("TRAINING TEACHER MODEL")
    print("="*60)
    
    # Create model
    teacher_model = create_lora_enhanced_teacher()
    teacher_model.compile(
        optimizer=Adam(learning_rate=OPTIMIZED_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Total parameters: {teacher_model.count_params():,}")
    
    # Count LoRA parameters
    lora_params = 0
    for layer in teacher_model.layers:
        if hasattr(layer, 'lora_A') and hasattr(layer, 'lora_B'):
            lora_params += layer.lora_A.shape[0] * layer.lora_A.shape[1]
            lora_params += layer.lora_B.shape[0] * layer.lora_B.shape[1]
    
    print(f"LoRA parameters: {lora_params:,} ({lora_params/teacher_model.count_params()*100:.2f}% of total)")
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    progress_logger = ProgressLogger()
    
    # Train
    history = teacher_model.fit(
        train_dataset,
        epochs=FINAL_EPOCHS,
        validation_data=test_dataset,
        callbacks=[early_stopping, progress_logger],
        verbose=1
    )
    
    # Save model
    teacher_model.save(f"{MODELS_DIR}/teacher_model.h5")
    teacher_model.save_weights(f"{MODELS_DIR}/teacher_model_weights.weights.h5")
    
    # Save metadata
    metadata = {
        'epochs_trained': FINAL_EPOCHS,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
        'class_names': CLASS_NAMES,
        'input_shape': INPUT_SHAPE,
        'num_classes': len(CLASS_NAMES),
        'model_type': 'teacher_ensemble'
    }
    
    with open(f"{TEACHER_OUTPUT_DIR}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save training history
    training_history = {
        'train_loss': [float(x) for x in history.history['loss']],
        'train_accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    
    with open(f"{TEACHER_OUTPUT_DIR}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save training curves
    save_training_curves(history.history, TEACHER_OUTPUT_DIR, "Teacher Model")
    
    print(f"\nTeacher model saved to {MODELS_DIR}")
    print(f"Training metadata saved to {TEACHER_OUTPUT_DIR}")
    
    return teacher_model, history


def train_student_model(teacher_model, train_dataset_teacher, test_dataset_student):
    """Train student model with knowledge distillation"""
    print("\n" + "="*60)
    print("TRAINING STUDENT MODEL WITH KNOWLEDGE DISTILLATION")
    print("="*60)
    
    # Create student model
    student_model = create_student_model()
    optimizer = Adam(learning_rate=OPTIMIZED_STUDENT_LR)
    student_model.compile(optimizer=optimizer, metrics=['accuracy'])
    optimizer.build(student_model.trainable_variables)
    
    print(f"Student model parameters: {student_model.count_params():,}")
    
    # Training history
    student_train_losses = []
    student_train_accuracies = []
    student_val_losses = []
    student_val_accuracies = []
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}")
        device = tf.test.gpu_device_name() if tf.config.list_physical_devices('GPU') else 'CPU'
        print(f"Training on: {device}")
        
        # Training
        total_loss_sum = 0
        accuracy_sum = 0
        steps = 0
        
        for x_teacher, y in train_dataset_teacher:
            total_loss, accuracy = train_step(student_model, teacher_model, optimizer, x_teacher, y)
            total_loss_sum += total_loss
            accuracy_sum += accuracy
            steps += 1
        
        avg_train_loss = total_loss_sum / steps
        avg_train_accuracy = accuracy_sum / steps
        
        student_train_losses.append(avg_train_loss.numpy())
        student_train_accuracies.append(avg_train_accuracy.numpy())
        
        # Validation
        val_loss_sum = 0
        val_accuracy_sum = 0
        val_steps = 0
        
        for x_student, y in test_dataset_student:
            student_logits = student_model(x_student, training=False)
            x_eff = preprocess_efficientnet(x_student)
            x_conv = preprocess_convnext(x_student)
            teacher_logits = teacher_model([x_eff, x_conv], training=False)
            val_loss = distillation_loss(y, student_logits, teacher_logits)
            predictions = tf.argmax(student_logits, axis=1, output_type=tf.int32)
            y = tf.cast(y, tf.int32)
            val_accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))
            val_loss_sum += val_loss
            val_accuracy_sum += val_accuracy
            val_steps += 1
            
            del x_eff, x_conv, teacher_logits, student_logits
            tf.keras.backend.clear_session()
        
        avg_val_loss = val_loss_sum / val_steps
        avg_val_accuracy = val_accuracy_sum / val_steps
        
        student_val_losses.append(avg_val_loss.numpy())
        student_val_accuracies.append(avg_val_accuracy.numpy())
        
        print(f"\nFinished Epoch {epoch + 1}/{EPOCHS}")
        print(f" - Training Loss: {avg_train_loss:.4f}")
        print(f" - Training Accuracy: {avg_train_accuracy:.4f}")
        print(f" - Validation Loss: {avg_val_loss:.4f}")
        print(f" - Validation Accuracy: {avg_val_accuracy:.4f}")
    
    # Save model
    student_model.save(f"{MODELS_DIR}/student_model.h5")
    
    # Save training history
    history_dict = {
        'loss': student_train_losses,
        'accuracy': student_train_accuracies,
        'val_loss': student_val_losses,
        'val_accuracy': student_val_accuracies
    }
    
    with open(f"{STUDENT_OUTPUT_DIR}/training_history.json", 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save training curves
    save_training_curves(history_dict, STUDENT_OUTPUT_DIR, "Student Model")
    
    print(f"\nStudent model saved to {MODELS_DIR}")
    print(f"Training history saved to {STUDENT_OUTPUT_DIR}")
    
    return student_model, history_dict

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("KNOWLEDGE DISTILLATION TRAINING - ORGANIZED VERSION")
    print("="*80)
    
    # Configure GPU
    configure_gpu()
    
    # Load dataset
    print("\nLoading dataset...")
    X_train_paths, X_test_paths, y_train, y_test, label_encoder = load_dataset()
    print(f"Training samples: {len(X_train_paths)}")
    print(f"Test samples: {len(X_test_paths)}")
    
    # Create datasets
    print("\nCreating TensorFlow datasets...")
    train_dataset_teacher, train_dataset_student, test_dataset_teacher, test_dataset_student = \
        create_datasets(X_train_paths, X_test_paths, y_train, y_test)
    
    # Train teacher model
    teacher_model, teacher_history = train_teacher_model(train_dataset_teacher, test_dataset_teacher)
    
    # Evaluate teacher model
    print("\n" + "="*60)
    print("EVALUATING TEACHER MODEL")
    print("="*60)
    teacher_results = comprehensive_evaluation(
        teacher_model, test_dataset_teacher,
        "Teacher Model", TEACHER_OUTPUT_DIR
    )
    
    # Freeze teacher model
    teacher_model.trainable = False
    
    # Train student model
    student_model, student_history = train_student_model(
        teacher_model, train_dataset_teacher, test_dataset_student
    )
    
    # Evaluate student model
    print("\n" + "="*60)
    print("EVALUATING STUDENT MODEL")
    print("="*60)
    student_results = comprehensive_evaluation(
        student_model, test_dataset_student,
        "Student Model", STUDENT_OUTPUT_DIR
    )
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)
    print(f"\nTeacher Model:")
    print(f"  Test Accuracy: {teacher_results['test_accuracy']:.4f}")
    print(f"  Test Loss: {teacher_results['test_loss']:.4f}")
    print(f"\nStudent Model:")
    print(f"  Test Accuracy: {student_results['test_accuracy']:.4f}")
    print(f"  Test Loss: {student_results['test_loss']:.4f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"  - Teacher outputs: {TEACHER_OUTPUT_DIR}")
    print(f"  - Student outputs: {STUDENT_OUTPUT_DIR}")
    print(f"  - Models: {MODELS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()


# =============================================================================
# COMMENTED CODE (Previously run sections - kept for reference)
# =============================================================================

"""
# Optuna hyperparameter optimization for teacher model
# (Commented out as optimization was already performed)

def realistic_teacher_objective(trial):
    try:
        tf.keras.backend.clear_session()
        
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        lora_rank = trial.suggest_categorical('lora_rank', [8, 16, 24])
        lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 48])
        attention_heads = trial.suggest_categorical('attention_heads', [4, 8, 12])
        window_size = trial.suggest_categorical('window_size', [2, 3, 4])
        global_tokens = trial.suggest_categorical('global_tokens', [8, 16, 24])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        dense_units = trial.suggest_categorical('dense_units', [128, 256, 384])
        dense_dropout = trial.suggest_float('dense_dropout', 0.3, 0.6)
        
        # [Rest of optimization code...]
        
        return best_val_accuracy
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        tf.keras.backend.clear_session()
        return 0.0


# Run optimization
# study = optuna.create_study(direction='maximize')
# study.optimize(realistic_teacher_objective, n_trials=3)
# print(f"Best Trial: {study.best_trial.number}")
# print(f"Best Validation Accuracy: {study.best_value:.4f}")
"""

"""
# Optuna hyperparameter optimization for student model
# (Commented out as optimization was already performed)

def student_hyperparameter_objective(trial):
    try:
        tf.keras.backend.clear_session()
        
        dense_units = trial.suggest_categorical('dense_units', [128, 256, 384, 512])
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        lora_rank = trial.suggest_categorical('lora_rank', [4, 8, 16, 24])
        lora_alpha = trial.suggest_categorical('lora_alpha', [8, 16, 32, 48])
        temperature = trial.suggest_float('temperature', 1.0, 10.0)
        distill_alpha = trial.suggest_float('distill_alpha', 0.5, 0.95)
        
        # [Rest of optimization code...]
        
        return best_val_accuracy
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        tf.keras.backend.clear_session()
        return 0.0


# Run optimization
# study_student = optuna.create_study(direction='maximize')
# study_student.optimize(student_hyperparameter_objective, n_trials=3)
# print(f"Best Trial: {study_student.best_trial.number}")
# print(f"Best Validation Accuracy: {study_student.best_value:.4f}")
"""
