# train_combined.py
import os
import json
import shutil
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# -------- CONFIG --------
DATA_DIR1 = "sorted_images"            # folder with 0..9,a..z subfolders
DATA_DIR2 = "Digit dataset 0-9"        # folder with 0..9 subfolders (or similar)
COMBINED_DIR = "combined_dataset"      # temporary merged dataset
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 40
MODEL_PATH = "char_digit_model.h5"
LABELS_PATH = "label_map.json"
SAMPLE_PER_CLASS = 5                    # for quick checking after training

# -------- helper: merge dataset directories into COMBINED_DIR --------
def merge_datasets(src_dirs, dest_dir):
    """
    Copy images from src_dirs into dest_dir/<class_name>/...
    Avoid overwriting by using unique filenames.
    """
    os.makedirs(dest_dir, exist_ok=True)
    total_copied = 0
    for src in src_dirs:
        if not os.path.exists(src):
            print(f"Warning: source dataset folder not found: {src} (skipping)")
            continue
        for class_name in os.listdir(src):
            class_path = os.path.join(src, class_name)
            if not os.path.isdir(class_path):
                continue
            dest_class_path = os.path.join(dest_dir, class_name)
            os.makedirs(dest_class_path, exist_ok=True)
            # copy files
            for fname in os.listdir(class_path):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
                    continue
                src_file = os.path.join(class_path, fname)
                # make a unique destination name to avoid collisions
                base, ext = os.path.splitext(fname)
                dest_file = os.path.join(dest_class_path, f"{base}_{random.randint(0,10**9)}{ext}")
                try:
                    shutil.copy2(src_file, dest_file)
                    total_copied += 1
                except Exception as e:
                    print(f"Failed to copy {src_file} -> {dest_file}: {e}")
    print(f"Merge complete. Total files copied: {total_copied}")

# -------- build datasets and model --------
def make_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    print("Creating datasets from:", data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size
    )
    return train_ds, val_ds

def build_model(num_classes, img_size=IMG_SIZE):
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=img_size + (1,))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="char_digit_cnn")
    return model

# -------- quick evaluation helpers --------
def load_image_for_model(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("L").resize(img_size)
    arr = np.array(img, dtype="float32") / 255.0
    # Force stroke bright: if background dark invert (this depends on your training)
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=-1)  # H,W,1
    arr = np.expand_dims(arr, axis=0)   # 1,H,W,1
    return arr

def sample_and_predict(model, label_map, combined_dir, n_per_class=1):
    # gather some files from the combined_dir
    samples = []
    for cls in os.listdir(combined_dir):
        cls_dir = os.path.join(combined_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff"))]
        if not files:
            continue
        for f in random.sample(files, min(n_per_class, len(files))):
            samples.append((cls, f))
    print(f"Will run quick predictions on {len(samples)} samples.")
    for cls, fpath in samples:
        arr = load_image_for_model(fpath)
        preds = model.predict(arr)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        lab = label_map.get(idx, str(idx))
        print(f"GT class-folder: {cls}   Predicted: {lab}   Confidence: {conf:.4f}   file: {fpath}")

# -------- main ----------
def main():
    # 1) Merge both datasets into one folder
    srcs = [DATA_DIR1, DATA_DIR2]
    print("Merging datasets:", srcs, "->", COMBINED_DIR)
    merge_datasets(srcs, COMBINED_DIR)

    # 2) Create TF datasets from combined folder
    train_ds, val_ds = make_datasets(COMBINED_DIR)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Detected classes:", class_names)
    print("Number of classes:", num_classes)
    if num_classes == 0:
        raise ValueError("No classes detected. Check your dataset structure and combined folder.")

    # Save label map (index -> class_name)
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)
    print(f"Label map saved to {LABELS_PATH}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # 3) Build and train model
    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr]
    )

    # 4) Save model
    model.save(MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # 5) Quick sanity predictions on a few samples
    print("Running quick sample predictions to sanity check model:")
    sample_and_predict(model, label_map, COMBINED_DIR, n_per_class= min(1, SAMPLE_PER_CLASS))

    print("Done.")

if __name__ == "__main__":
    main()
