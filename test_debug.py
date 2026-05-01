import sys
import traceback

try:
    import os
    print("Step 1: os imported", flush=True)
    
    import numpy as np
    print("Step 2: numpy imported", flush=True)
    
    import cv2
    print("Step 3: cv2 imported", flush=True)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Step 4: matplotlib imported", flush=True)
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    print("Step 5: sklearn imported", flush=True)
    
    import tensorflow as tf
    print("Step 6: tensorflow imported:", tf.__version__, flush=True)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Conv2D, Flatten, Dropout, MaxPooling2D,
        BatchNormalization, GlobalAveragePooling2D
    )
    print("Step 7: keras layers imported", flush=True)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("Step 8: ImageDataGenerator imported", flush=True)
    
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    print("Step 9: callbacks imported", flush=True)
    
    from tensorflow.keras.optimizers import Adam
    print("Step 10: Adam imported", flush=True)

    print("All imports successful!", flush=True)
    
    # Now test dataset loading
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "fer2013", "train")
    TEST_DIR = os.path.join(BASE_DIR, "dataset", "fer2013", "test")
    
    print(f"Train dir exists: {os.path.exists(TRAIN_DIR)}", flush=True)
    print(f"Test dir exists: {os.path.exists(TEST_DIR)}", flush=True)
    
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Try loading just a few images
    path = os.path.join(TRAIN_DIR, 'angry')
    files = os.listdir(path)[:5]
    print(f"First 5 angry files: {files}", flush=True)
    
    img = cv2.imread(os.path.join(path, files[0]), cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {img.shape}", flush=True)
    
    print("BASIC TEST PASSED!", flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
