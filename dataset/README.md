Unzip and save data from Roboflow in this directory.

- Metadata dataset should be saved in folder `metadata`
- Content dataset should be saved in folder `content`

```
dataset/
  metadata/
    images/    # Cropped Student ID and Exam ID images
    labels/    # YOLO label files for metadata images
    data.yaml  # Label information for metadata model
  content/
    images/    # Cropped content images (Content11, Content12, ...)
    labels/    # YOLO label files for content images
    data.yaml  # Label information for content model
  data
```
