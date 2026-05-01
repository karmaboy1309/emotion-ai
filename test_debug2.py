import sys
import traceback

try:
    import os
    print("os OK", flush=True)
    
    import tensorflow as tf
    print("TF:", tf.__version__, flush=True)
    
    import numpy as np
    print("numpy OK", flush=True)

    import cv2
    print("cv2 OK", flush=True)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("matplotlib OK", flush=True)
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    print("sklearn OK", flush=True)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Conv2D, Flatten, Dropout, MaxPooling2D,
        BatchNormalization, GlobalAveragePooling2D
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.optimizers import Adam
    print("All keras imports OK", flush=True)
    
    print("ALL IMPORTS SUCCESS!", flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
