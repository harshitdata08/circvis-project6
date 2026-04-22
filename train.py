import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


BACKBONES = {
    "mobilenet": tf.keras.applications.MobileNetV2,
    "resnet": tf.keras.applications.ResNet50,
    "efficientnet": tf.keras.applications.EfficientNetB0,
}

PREPROCESS = {
    "mobilenet": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet": tf.keras.applications.resnet.preprocess_input,
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIRCVIS waste classifier")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="efficientnet", choices=list(BACKBONES.keys()))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models")
    return parser.parse_args()


def load_datasets(data_dir, img_size, batch_size):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(model_name, img_size, num_classes, learning_rate, fine_tune=False):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = PREPROCESS[model_name](x)

    base_model = BACKBONES[model_name](
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = fine_tune

    x = base_model(x, training=fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def save_training_plot(history, output_dir):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=200)
    plt.close()


def evaluate_and_save(model, test_ds, class_names, output_dir):
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, args.img_size, args.batch_size)
    model = build_model(args.model, args.img_size, len(class_names), args.learning_rate, fine_tune=args.fine_tune)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir, "best_model.keras"), save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2, monitor="val_loss"),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    save_training_plot(history, args.output_dir)
    evaluate_and_save(model, test_ds, class_names, args.output_dir)

    with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f)

    print("Training complete. Files saved in:", args.output_dir)


if __name__ == "__main__":
    main()
