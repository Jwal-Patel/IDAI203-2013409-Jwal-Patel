"""
train_model_improved.py
Improved training script with:
- Reproducible train/val/test splits
- Data augmentation (including brightness)
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Initial feature-extraction phase + fine-tuning phase
- Save model, class_indices.json, and training history (history.json)
- Produce and save training plots and confusion matrix image
Adjust paths and parameters as needed.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import pathlib
import random

# -------------------- USER CONFIG --------------------
DATA_DIR = "dataset"  # directory with class subfolders
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
INITIAL_EPOCHS = 8
FINETUNE_EPOCHS = 8
FINE_TUNE_AT = 100  # number of layers from the end to keep trainable during fine-tune
MODEL_SAVE_PATH = "waste_model_improved.h5"
CLASS_INDICES_PATH = "class_indices.json"
HISTORY_PATH = "history.json"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
# -----------------------------------------------------

# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Create train/val/test split reproducibly by listing files
def create_splits(data_dir, val_split=0.15, test_split=0.10, seed=SEED):
    data_dir = pathlib.Path(data_dir)
    train_files = []
    val_files = []
    test_files = []
    classes = []
    for cls_dir in sorted([d for d in data_dir.iterdir() if d.is_dir()]):
        cls_name = cls_dir.name
        classes.append(cls_name)
        files = list(cls_dir.glob("*"))
        files = [str(p) for p in files if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        files = sorted(files)
        random.Random(seed).shuffle(files)
        n = len(files)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test = files[:n_test]
        val = files[n_test:n_test + n_val]
        train = files[n_test + n_val:]
        train_files += train
        val_files += val
        test_files += test
    return classes, train_files, val_files, test_files

classes, train_files, val_files, test_files = create_splits(DATA_DIR)
print(f"Detected classes: {classes}")
print(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}, Test samples: {len(test_files)}")

# Save class indices mapping
class_indices = {i: cls for i, cls in enumerate(classes)}
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f, indent=2)

# We will use ImageDataGenerator with flow_from_directory, so create a temporary directory structure for splits
def make_split_dirs(base_dir="dataset_splits"):
    base = pathlib.Path(base_dir)
    if base.exists():
        # optionally clear; here we recreate to be safe
        pass
    base.mkdir(exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in classes:
            d = base / split / cls
            d.mkdir(parents=True, exist_ok=True)
    # copy files
    import shutil
    for p in train_files:
        cls = pathlib.Path(p).parent.name
        shutil.copy(p, base / "train" / cls / pathlib.Path(p).name)
    for p in val_files:
        cls = pathlib.Path(p).parent.name
        shutil.copy(p, base / "val" / cls / pathlib.Path(p).name)
    for p in test_files:
        cls = pathlib.Path(p).parent.name
        shutil.copy(p, base / "test" / cls / pathlib.Path(p).name)
    return str(base)

SPLIT_DIR = make_split_dirs()

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=(0.6,1.2),
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_generator = test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

# Build model (MobileNetV2 baseline)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
early = EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)

history1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, early, reduce_lr]
)

# Fine-tuning
base_model.trainable = True
# freeze early layers
if FINE_TUNE_AT is not None and FINE_TUNE_AT < len(base_model.layers):
    for layer in base_model.layers[:-FINE_TUNE_AT]:
        layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS + FINETUNE_EPOCHS,
    initial_epoch=history1.epoch[-1] + 1 if hasattr(history1, "epoch") and len(history1.epoch)>0 else 0,
    validation_data=val_generator,
    callbacks=[checkpoint, early, reduce_lr]
)

# Merge history
def merge_history(h1, h2):
    history = {}
    for k in h1.history.keys():
        history[k] = h1.history[k] + h2.history.get(k, [])
    for k in h2.history.keys():
        if k not in history:
            history[k] = h2.history[k]
    return history

history = merge_history(history1, history2)

# Save final model and history
model.save(MODEL_SAVE_PATH, include_optimizer=False)
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)

# Plot training curves
plt.figure()
plt.plot(history.get("accuracy", []))
plt.plot(history.get("val_accuracy", []))
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train","val"])
plt.savefig(os.path.join(PLOTS_DIR, "accuracy.png"))
plt.close()

plt.figure()
plt.plot(history.get("loss", []))
plt.plot(history.get("val_loss", []))
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train","val"])
plt.savefig(os.path.join(PLOTS_DIR, "loss.png"))
plt.close()

# Evaluate on test set and save confusion matrix
# Ensure test_generator has shuffle=False
test_steps = test_generator.samples // test_generator.batch_size + int(test_generator.samples % test_generator.batch_size != 0)
preds = model.predict(test_generator, steps=test_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# classification report
report = classification_report(y_true, y_pred, target_names=classes, digits=4)
print(report)
with open(os.path.join(PLOTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

plot_confusion_matrix(cm, classes)
print("Training complete. Model, history and plots saved.")
