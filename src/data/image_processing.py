import cv2
import numpy as np
import os
import sys
import argparse
from pdf2image import convert_from_path

# Constants
# Image size
IMG_WIDTH = 2480
IMG_HEIGHT = 3508

# Align reference image
REF_IMG_GRAY = cv2.imread("./assets/references/reference.jpg", cv2.IMREAD_GRAYSCALE)
REF_IMG_GRAY = cv2.resize(REF_IMG_GRAY, (IMG_WIDTH, IMG_HEIGHT))

# Lookup table for gamma correction
gamma = 2.2
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

# Small image size for padding
SMALL_IMG_WIDTH = 640
SMALL_IMG_HEIGHT = 640

# Split coordinates
STUDENT_ID_CONTOUR_1_COORD = (180, 352)
STUDENT_ID_CONTOUR_2_COORD = (666, 1129)

EXAM_ID_CONTOUR_1_COORD = (687, 352)
EXAM_ID_CONTOUR_2_COORD = (878, 1129)

CONTENT_11_CONTOUR_1_COORD = (230, 1265)
CONTENT_11_CONTOUR_2_COORD = (710, 1605)

CONTENT_12_CONTOUR_1_COORD = (750, 1265)
CONTENT_12_CONTOUR_2_COORD = (1230, 1605)

CONTENT_13_CONTOUR_1_COORD = (1270, 1265)
CONTENT_13_CONTOUR_2_COORD = (1750, 1605)

CONTENT_14_CONTOUR_1_COORD = (1790, 1265)
CONTENT_14_CONTOUR_2_COORD = (2270, 1605)

CONTENT_21_CONTOUR_1_COORD = (230, 1685)
CONTENT_21_CONTOUR_2_COORD = (710, 2180)

CONTENT_22_CONTOUR_1_COORD = (750, 1685)
CONTENT_22_CONTOUR_2_COORD = (1230, 2180)

CONTENT_23_CONTOUR_1_COORD = (1270, 1685)
CONTENT_23_CONTOUR_2_COORD = (1750, 2180)

CONTENT_24_CONTOUR_1_COORD = (1790, 1685)
CONTENT_24_CONTOUR_2_COORD = (2270, 2180)


"""
Reads a PDF file and splits each page into a resized JPEG image.

Args:
    pdf_path (str): Path to the PDF file.

Returns:
    List of resized images as numpy arrays.
"""
def pdf2img(pdf_path):
    pil_images = convert_from_path(pdf_path)
    images = []

    for pil_img in pil_images:
        open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        resized_img = cv2.resize(open_cv_image, (IMG_WIDTH,IMG_HEIGHT))
        images.append(resized_img)
    
    return images

"""
Aligns the scanned image to the reference image using SIFT and FLANN.

Args:
    scan_img: The scanned image to be aligned.
    ref_img_gray: The reference image in grayscale.
    target_size: The target size for the aligned image.

Returns:
    return aligned_img: The aligned image.
"""
def align_image(scan_img):
    scan_img_gray = cv2.cvtColor(scan_img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()
    scan_keypoints, scan_descriptors = sift.detectAndCompute(scan_img_gray, None)
    ref_keypoints, ref_descriptors = sift.detectAndCompute(REF_IMG_GRAY, None)

    if scan_descriptors is None or ref_descriptors is None:
        raise ValueError("Failed to compute SIFT descriptors.")

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches using k-NN and Lowe ratio
    knn_matches = flann.knnMatch(scan_descriptors, ref_descriptors, k=2)    
    good_matches = set()
    threshold = 0.2
    while len(good_matches) < 15 and threshold < 1.0:
        good_matches.clear()
        for m, n in knn_matches:
            if m.distance < threshold * n.distance:
                good_matches.add(m)
        if threshold < 1:
            threshold += 0.1
        else:
            raise ValueError("Can not find at least 15 good matches.")
    print(f"Good matches found: {len(good_matches)} with threshold {threshold - 0.1}")

    # Extract matched keypoints locations
    points1 = np.float32([scan_keypoints[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches])

    # Compute homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)
    if h is None:
        raise ValueError("Homography estimation failed.")

    # Align
    aligned_img = cv2.warpPerspective(scan_img, h, (IMG_WIDTH,IMG_HEIGHT), flags=cv2.INTER_LINEAR)
    return aligned_img


"""
Function to paddding image to 640x640

Args: 
    image: The image to resize and padding.

Returns: 
    The padded image.

"""
def padding_image(image):
    h, w = image.shape[:2]
    scale = 640 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    image = cv2.resize(image, (new_w, new_h))

    pad_w, pad_h = 640 - new_w, 640 - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Pipeline to split image
def pipeline(image):
    # Align image
    aligned_image = align_image(image)
    image = cv2.LUT(aligned_image, lookUpTable)

    # Split image
    studentId = image[STUDENT_ID_CONTOUR_1_COORD[1]:STUDENT_ID_CONTOUR_2_COORD[1], 
                      STUDENT_ID_CONTOUR_1_COORD[0]:STUDENT_ID_CONTOUR_2_COORD[0]]
    examId = image[EXAM_ID_CONTOUR_1_COORD[1]:EXAM_ID_CONTOUR_2_COORD[1], 
                   EXAM_ID_CONTOUR_1_COORD[0]:EXAM_ID_CONTOUR_2_COORD[0]]

    content11 = image[CONTENT_11_CONTOUR_1_COORD[1]:CONTENT_11_CONTOUR_2_COORD[1], 
                      CONTENT_11_CONTOUR_1_COORD[0]: CONTENT_11_CONTOUR_2_COORD[0]]
    content12 = image[CONTENT_12_CONTOUR_1_COORD[1]:CONTENT_12_CONTOUR_2_COORD[1], 
                      CONTENT_12_CONTOUR_1_COORD[0]: CONTENT_12_CONTOUR_2_COORD[0]]
    content13 = image[CONTENT_13_CONTOUR_1_COORD[1]:CONTENT_13_CONTOUR_2_COORD[1], 
                      CONTENT_13_CONTOUR_1_COORD[0]: CONTENT_13_CONTOUR_2_COORD[0]]
    # content14 = image[CONTENT_14_CONTOUR_1_COORD[1]:CONTENT_14_CONTOUR_2_COORD[1], 
    #                   CONTENT_14_CONTOUR_1_COORD[0]: CONTENT_14_CONTOUR_2_COORD[0]]
    
    content21 = image[CONTENT_21_CONTOUR_1_COORD[1]:CONTENT_21_CONTOUR_2_COORD[1], 
                      CONTENT_21_CONTOUR_1_COORD[0]: CONTENT_21_CONTOUR_2_COORD[0]]
    content22 = image[CONTENT_22_CONTOUR_1_COORD[1]:CONTENT_22_CONTOUR_2_COORD[1], 
                      CONTENT_22_CONTOUR_1_COORD[0]: CONTENT_22_CONTOUR_2_COORD[0]]
    content23 = image[CONTENT_23_CONTOUR_1_COORD[1]:CONTENT_23_CONTOUR_2_COORD[1], 
                      CONTENT_23_CONTOUR_1_COORD[0]: CONTENT_23_CONTOUR_2_COORD[0]]
    # content24 = image[CONTENT_24_CONTOUR_1_COORD[1]:CONTENT_24_CONTOUR_2_COORD[1], 
    #                   CONTENT_24_CONTOUR_1_COORD[0]: CONTENT_24_CONTOUR_2_COORD[0]]

    # Padding split images
    studentId = padding_image(studentId)
    examId = padding_image(examId)
    content11 = padding_image(content11)
    content12 = padding_image(content12)
    content13 = padding_image(content13)
    # content14 = padding_image(content14)
    content21 = padding_image(content21)
    content22 = padding_image(content22)
    content23 = padding_image(content23)
    # content24 = padding_image(content24)

    result = {"studentId": studentId, "examId": examId, 
              "content11": content11, "content12": content12, "content13": content13, 
              "content21": content21, "content22": content22, "content23": content23}
    
    return result

# Main execute
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split scanned images from PDF")
    parser.add_argument(
        "-i", "--input_path",
        help="Path to the input file"
    )
    args = parser.parse_args()

    pdf_path = args.input_path
    if not pdf_path.lower().endswith('.pdf'):
        print("Error: Provided file is not a PDF.")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join("./assets/data", base_name)
    metadata_folder = os.path.join(output_folder, "metadata/images")
    content_folder = os.path.join(output_folder, "content/images")
    for folder in [output_folder, metadata_folder, content_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print("Converting PDF pages to images...")
    images = pdf2img(pdf_path)
    for idx, image in enumerate(images):
        image_base_name = f"{base_name}_page_{idx + 1}"
        
        print(f"Processing {image_base_name}...")
        parts = pipeline(image)

        cv2.imwrite(os.path.join(metadata_folder, image_base_name + "_studentId.jpg"), parts["studentId"])
        cv2.imwrite(os.path.join(metadata_folder, image_base_name + "_examId.jpg"), parts["examId"])
        cv2.imwrite(os.path.join(content_folder, image_base_name + "_content11.jpg"), parts["content11"])
        cv2.imwrite(os.path.join(content_folder, image_base_name + "_content12.jpg"), parts["content12"])
        cv2.imwrite(os.path.join(content_folder, image_base_name + "_content13.jpg"), parts["content13"])
        cv2.imwrite(os.path.join(content_folder, image_base_name + "_content21.jpg"), parts["content21"])
        cv2.imwrite(os.path.join(content_folder, image_base_name + "_content22.jpg"), parts["content22"])
        cv2.imwrite(os.path.join(content_folder, image_base_name + "_content23.jpg"), parts["content23"])
        
        print(f"Saved split parts for {image_base_name}.\n")
