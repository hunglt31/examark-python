import onnxruntime as ort
import numpy as np
import cv2

# Load model
model_path = "./assets/models/metadata_model_plus.onnx"
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# Load and preprocess 2 images
img1 = cv2.imread("./assets/data/unlabel_metadata/Mau_scan (1)_studentID.png")
img2 = cv2.imread("./assets/data/unlabel_metadata/Mau_scan (1)_examID.png")

def non_max_suppression(boxes, confidences, score_thr=1e-3, iou_thr=0.5):
    """
    boxes: list of [x, y, w, h]
    confidences: list of float scores
    returns: list of kept indices (ints)
    """
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_thr, iou_thr)
    # if it comes back as a (N,1) array, flatten it:
    if isinstance(idxs, np.ndarray):
        idxs = idxs.flatten().tolist()
    return idxs

imgs = [img1, img2]
input_blob = np.stack(imgs, axis=0) 

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_blob})

boxes = outputs[0] 
scores = outputs[1]

print("Output shapes:", boxes.shape, scores.shape) 

for img_idx in range(boxes.shape[0]):
    img_boxes = boxes[img_idx]        # shape: (8400, 4)
    img_scores = scores[img_idx]      # shape: (8400, 2)

    boxes_list, confidences, class_ids = [], [], []
    class0_count = class1_count = 0

    for box, score in zip(img_boxes, img_scores):
        x_c, y_c, w, h = box
        class0_score, class1_score = score

        if class1_score > 0.5:
            boxes_list.append([x_c - w / 2, y_c - h / 2, w, h])
            confidences.append(class1_score)
            class_ids.append(1)
            class1_count += 1

        if class0_score > 0.5:
            boxes_list.append([x_c - w / 2, y_c - h / 2, w, h])
            confidences.append(class0_score)
            class_ids.append(0)
            class0_count += 1

    print(f"Image {img_idx}: {class1_count} class 1 boxes, {class0_count} class 0 boxes")
    keep = non_max_suppression(boxes_list, confidences)
    print(f"Image {img_idx}: {len(keep)} boxes kept after NMS")

    img = img1 if img_idx == 0 else img2
    for i in keep:
        x, y, w, h = map(int, boxes_list[i])
        color = (255, 0, 0) if class_ids[i] == 1 else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)



# Save results
cv2.imwrite("img1_output.png", img1)
cv2.imwrite("img2_output.png", img2) 