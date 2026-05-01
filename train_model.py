"""
Improved Facial Emotion Recognition Model Training Script
=========================================================
This script trains a significantly improved CNN model for facial emotion recognition
using the FER-2013 dataset with:
- Deeper architecture with BatchNormalization
- Data augmentation for better generalization
- Class weight balancing (handles imbalanced classes like 'disgust')
- Learning rate scheduling & early stopping
- Comprehensive evaluation with confusion matrix & classification report
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Conv2D, Flatten, Dropout, MaxPooling2D,
    BatchNormalization, GlobalAveragePooling2D, Input, SeparableConv2D, Activation, add
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
import time

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "fer2013", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "fer2013", "test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "emotion_model.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "training_results")

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 60  # Will use early stopping, so this is max epochs
NUM_CLASSES = 7

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

print("=" * 60)
print("  FACIAL EMOTION RECOGNITION - MODEL TRAINING")
print("=" * 60)

# ============================================================
# Step 1: Load Data
# ============================================================
print("\n[1/6] Loading dataset...")

def load_data(data_dir):
    data = []
    labels = []
    for label in emotion_labels:
        path = os.path.join(data_dir, label)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping...")
            continue
        files = os.listdir(path)
        print(f"  {label}: {len(files)} images")
        for img_file in files:
            img_path = os.path.join(path, img_file)
            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(label)
            except Exception as e:
                continue
    return np.array(data), np.array(labels)

print("\nTraining data:")
X_train, y_train_raw = load_data(TRAIN_DIR)
print(f"\nTest data:")
X_test, y_test_raw = load_data(TEST_DIR)

print(f"\n  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")

# ============================================================
# Step 2: Preprocess Data
# ============================================================
print("\n[2/6] Preprocessing data...")

# Normalize images (0-255 to 0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for model input (add channel dimension)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# Compute class weights to handle imbalanced dataset
# (disgust has only ~436 samples vs happy with ~7215)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("  Class weights (to handle imbalance):")
for idx, label in enumerate(emotion_labels):
    print(f"    {label}: {class_weight_dict[idx]:.3f}")

print("  Data preprocessed successfully!")

# ============================================================
# Step 3: Data Augmentation
# ============================================================
print("\n[3/6] Setting up data augmentation...")

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1,
    fill_mode='nearest'
)

train_datagen.fit(X_train)
print("  Data augmentation configured!")

# ============================================================
# Step 4: Build Improved Model
# ============================================================
print("\n[4/6] Building improved Mini-Xception architecture...")

def build_mini_xception(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    input_layer = Input(shape=input_shape)
    
    # Base convolution
    x = Conv2D(8, (3, 3), padding='same', use_bias=False)(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual Module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])
    
    # Residual Module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])
    
    # Residual Module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])
    
    # Residual Module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])
    
    # Final Classification
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

model = build_mini_xception()

# Compile with Adam optimizer
optimizer = Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

total_params = model.count_params()
print(f"\n  Total parameters: {total_params:,}")

# ============================================================
# Step 5: Train the Model
# ============================================================
print("\n[5/6] Training model...")
print(f"  Epochs: {EPOCHS} (max, with early stopping)")
print(f"  Batch size: {BATCH_SIZE}")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

start_time = time.time()

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n  Training completed in {training_time/60:.1f} minutes")

# ============================================================
# Step 6: Evaluate the Model
# ============================================================
print("\n[6/6] Evaluating model...")

# Load best model
from tensorflow.keras.models import load_model
best_model = load_model(MODEL_SAVE_PATH)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\n{'='*60}")
print(f"  FINAL MODEL ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*60}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=emotion_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# ============================================================
# Save Training Plots
# ============================================================
print("\nSaving training plots...")

# Plot Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {os.path.join(RESULTS_DIR, 'training_history.png')}")

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(conf_matrix.shape[1]),
    yticks=np.arange(conf_matrix.shape[0]),
    xticklabels=emotion_labels,
    yticklabels=emotion_labels,
    ylabel='True Label',
    xlabel='Predicted Label',
    title='Confusion Matrix'
)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, format(conf_matrix[i, j], 'd'),
                ha='center', va='center',
                color='white' if conf_matrix[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")

# Save results summary
with open(os.path.join(RESULTS_DIR, 'training_summary.txt'), 'w') as f:
    f.write("FACIAL EMOTION RECOGNITION - TRAINING SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Training Time: {training_time/60:.1f} minutes\n")
    f.write(f"Total Epochs Run: {len(history.history['accuracy'])}\n")
    f.write(f"Final Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Best Val Accuracy: {max(history.history['val_accuracy'])*100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_classes, target_names=emotion_labels))

print(f"  Saved: {os.path.join(RESULTS_DIR, 'training_summary.txt')}")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print(f"  Model saved to: {MODEL_SAVE_PATH}")
print(f"  Accuracy: {accuracy*100:.2f}%")
print("=" * 60)
