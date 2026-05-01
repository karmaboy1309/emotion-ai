import sys
print("Python:", sys.version)

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TF import error:", e)

try:
    from tensorflow.keras.models import Sequential
    print("Sequential: OK")
except Exception as e:
    print("Sequential error:", e)

try:
    from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
    print("BatchNormalization: OK")
except Exception as e:
    print("BatchNorm error:", e)

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("ImageDataGenerator: OK")
except Exception as e:
    print("ImageDataGen error:", e)

try:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print("Callbacks: OK")
except Exception as e:
    print("Callbacks error:", e)

try:
    from sklearn.utils.class_weight import compute_class_weight
    print("class_weight: OK")
except Exception as e:
    print("class_weight error:", e)

print("All imports tested!")
