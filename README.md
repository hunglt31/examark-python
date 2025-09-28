# Examark Python Project

## Overview

This project provides a pipeline for template matching, image processing, dataset creation, model training, and automated labeling for computer vision tasks.

---

## Data Storage Structure

- Manually put pdf files in `assets/pdf/` directory.
- Trained models are saved in the `assets/models/` directory by default.
- Processed data is stored in the `assets/data/` directory.

Each PDF you process creates a subfolder named after the PDF file (without extension):

```
assets/data/
  pdf_1/
    metadata/
      images/    # Cropped Student ID and Exam ID images
      labels/    # YOLO label files for metadata images
    content/
      images/    # Cropped content images (Content11, Content12, ...)
      labels/    # YOLO label files for content images
  pdf_2/
    metadata/
      images/
      labels/
    content/
      images/
      labels/
  ...
  pdf_n/
    metadata/
      images/
      labels/
    content/
      images/
      labels/
```

---

## Workflow

### 1. Data Preparation

- **Template Matching:**  
  Run `template_matching.py` to detect and extract black squares from reference image.

- **Configure Coordinates:**  
  Edit `image_processing.py` to set the coordinates for regions.

- **Image Processing:**  
  Run `image_processing.py` to process images and create the dataset.

---

### 2. Model Training

- **Training:**  
  Use `train.py` to train model on the prepared dataset.

- **Add Layers:**  
  If need to modify the model architecture, use the `add_layers_metadata_model.py` and `add_layers_content_model.py`.
  We will build a TensorRT engine later for deployment.

---

### 3. Automated Labeling

- **Label Dataset:**  
  After training, run `label_dataset.py` to automatically label all subfolders in `./assets/data` using the trained model.

- **Roboflow Upload:**  
  Upload the labeled data to Roboflow for further inspection or validation.

---

## Example Usage

```bash
# 1. Run template matching and configure coordinates
python3 ./src/data/template_matching.py

# 2. Process images
python3 ./src/data/image_processing.py -i <pdf_file_path>

# 3. Train the model
python3 ./src/model/train.py

# 4. Add layers to the model
# Metadata model
python3 ./src/model/add_layers_metadata_model.py
# Content model
python3 ./src/model/add_layers_content_model.py

# 5. Label the dataset using the trained model
python3 .src/data/label_dataset.py -i ./assets/data

# 6. Upload labeled data to Roboflow for checking
```

---

## Requirements

- Python 3.10.12
- OpenCV
- Ultralytics YOLO
- ONNX

---
