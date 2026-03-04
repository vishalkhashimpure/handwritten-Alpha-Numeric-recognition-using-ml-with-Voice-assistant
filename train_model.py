# train_model.py

import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === CONFIG ===
DATA_DIR = r"sorted_images"   # path where you extracted sorted_images.zip
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10                       # increase if you have time/GPU
MODEL_PATH = "char_digit_model.h5"
LABELS_PATH = "label_map.json"

def main():
    # 1. Load dataset from directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Number of classes:", num_classes)

    # Save label map so inference script knows index -> character
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)

    # 2. Performance tweaks (prefetch & cache)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Build the model (simple CNN)
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=IMG_SIZE + (1,)),  # grayscale

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # 4. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 5. Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label map saved to {LABELS_PATH}")

if __name__ == "__main__":
    main()



------------------------------ another dataset ----------------------------

# train_model.py

import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === CONFIG ===
DATA_DIR = r"digit dataset 0-9"   # path where you extracted sorted_images.zip
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10                       # increase if you have time/GPU
MODEL_PATH = "char_digit_model.h5"
LABELS_PATH = "label_map.json"

def main():
    # 1. Load dataset from directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Number of classes:", num_classes)

    # Save label map so inference script knows index -> character
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)

    # 2. Performance tweaks (prefetch & cache)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Build the model (simple CNN)
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=IMG_SIZE + (1,)),  # grayscale

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # 4. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 5. Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label map saved to {LABELS_PATH}")

if __name__ == "__main__":
    main()
