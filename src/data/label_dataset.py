import os
import argparse
from ultralytics import YOLO

def process_folder(model, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(input_folder, file)
            results = model(path)

            boxes = results[0].boxes
            if len(boxes) > 0:
                # pick the best (highest confidence) box
                best_box = boxes[boxes.conf.argmax().item()]
                out_file = file.rsplit(".", 1)[0] + ".txt"
                out_path = os.path.join(output_folder, out_file)

                with open(out_path, "w") as f:
                    cls = int(best_box.cls.item())
                    conf = best_box.conf.item()
                    xywh = best_box.xywhn.view(-1).tolist()
                    f.write(f"{cls} {' '.join(map(str, xywh))} {conf}\n")


def main():
    parser = argparse.ArgumentParser(description="Label dataset with YOLO models")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to dataset root folder")
    args = parser.parse_args()

    # load models once
    metadata_model = YOLO("./models/metadata/weights/best.pt")
    content_model = YOLO("./models/content/weights/best.pt")

    # iterate over all subfolders of input_dir
    for subfolder in os.listdir(args.input_dir):
        folder_path = os.path.join(args.input_dir, subfolder)
        if not os.path.isdir(folder_path):
            continue

        # metadata
        metadata_in = os.path.join(folder_path, "metadata/images")
        metadata_out = os.path.join(folder_path, "metadata/labels")
        if os.path.isdir(metadata_in):
            print(f"Processing metadata in {metadata_in}")
            process_folder(metadata_model, metadata_in, metadata_out)

        # content
        content_in = os.path.join(folder_path, "content/images")
        content_out = os.path.join(folder_path, "content/labels")
        if os.path.isdir(content_in):
            print(f"Processing content in {content_in}")
            process_folder(content_model, content_in, content_out)


if __name__ == "__main__":
    main()
