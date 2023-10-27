import json
import os

def convert_to_yolo(json_file, class_mapping, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
        arrayData = data.items()

    for entry in arrayData:
        bboxes = entry[1]["bboxes"]
        labels = entry[1]["labels"]
        notGest = "no_gesture"
        image_path = f"Object Detection/images/{entry[0]}.jpg"  # If "image_path" key is present, it will be used, otherwise an empty string.
        if not image_path:
            print("Warning: Missing 'image_path' in the entry. Skipping.")
            continue

        image = Image.open(image_path)
        width, height = image.size

        yolo_filename = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", ".txt"))

        with open(yolo_filename, "w") as yolo_file:
            for bbox, label in zip(bboxes, labels):
                if label in class_mapping:
                    class_id = class_mapping[label]
                else:
                    print(f"Warning: Label '{label}' not found in class mapping. Skipping. {entry[0]}")
                    continue

                x_center = bbox[0]
                y_center = bbox[1]
                bbox_width = bbox[2]
                bbox_height = bbox[3]

                x = x_center
                y = y_center
                w = bbox_width
                h = bbox_height

                yolo_file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

class_mapping = {
            "call": 0,
            "dislike": 1,
            "fist": 2,
            "four": 3,
            "like": 4,
            "mute": 5,
            "ok": 6,
            "one": 7,
            "palm": 8,
            "peace": 9,
            "peace_inverted": 10,
            "rock": 11,
            "stop": 12,
            "stop_inverted": 13,
            "three": 14,
            "three2": 15,
            "two_up": 16,
            "two_up_inverted": 17,
            # Add more class mappings as needed
        }

for gesture in class_mapping:
    if __name__ == "__main__":
        
        json_file = f"Object Detection/labels/{gesture}.json"
        output_dir = f"Object Detection/newLabels"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        from PIL import Image
        convert_to_yolo(json_file, class_mapping, output_dir)
