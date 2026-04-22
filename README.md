# CIRCVIS: Context-Aware Waste Classification for Circular Cities

CIRCVIS is a student-friendly deep learning project for waste image classification. It uses transfer learning with TensorFlow/Keras and supports multiple backbone models such as MobileNetV2, ResNet50, and EfficientNetB0.

## Project Structure

- `train.py` - Train the model on an image dataset arranged by class folders
- `app.py` - Streamlit web app for demo prediction
- `predict.py` - Command-line inference script
- `requirements.txt` - Python dependencies
- `assets/` - Optional screenshots, logo, confusion matrix exports
- `data/` - Put dataset here
- `models/` - Saved models will be written here

## Recommended Dataset Layout

```text
circvis_project/
  data/
    train/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
    val/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
    test/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
```

You can adapt the classes if you use RealWaste, TACO subsets, or your own collected data.

## Installation

```bash
pip install -r requirements.txt
```

## Train Example

```bash
python train.py --data_dir data --model efficientnet --epochs 10 --img_size 224 --batch_size 32
```

## Run Demo App

```bash
streamlit run app.py
```

## Predict One Image

```bash
python predict.py --image sample.jpg --model_path models/best_model.keras
```

## Suggested Submission Deliverables

1. Report / PDF
2. Hosted project website
3. Source code bundle
4. Screenshots of training curves
5. Confusion matrix image
6. Demo screenshots from Streamlit app

## Notes

- For best student results, start with `mobilenet` or `efficientnet`.
- Use image augmentation to improve robustness.
- Keep train/val/test split fixed for fair comparison.
- Save confusion matrix and classification report for viva.
