import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image

PREPROCESS = {
    "mobilenet": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet": tf.keras.applications.resnet.preprocess_input,
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict waste class from one image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, default="models/class_names.json")
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=list(PREPROCESS.keys()))
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    with open(args.labels_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    image = Image.open(args.image).convert("RGB").resize((args.img_size, args.img_size))
    arr = np.array(image, dtype=np.float32)
    arr = PREPROCESS[args.backbone](arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    print("Predicted class:", class_names[top_idx])
    print("Confidence:", round(float(probs[top_idx]) * 100, 2), "%")

    print("\nAll probabilities:")
    for label, prob in sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True):
        print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()
