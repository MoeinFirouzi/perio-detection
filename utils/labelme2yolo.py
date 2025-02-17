import json
import os


def convert_labelme_polygons_to_yolo(json_dir, output_dir, class_names):
    """
    Converts LabelMe JSON polygon annotations to YOLOv8 format (segmentation).

    Parameters:
    - json_dir (str): Directory containing LabelMe JSON files.
    - output_dir (str): Directory to save YOLO label `.txt` files.
    - class_names (list): List of class names in the dataset.

    Example usage:
    convert_labelme_polygons_to_yolo("path/to/json", "path/to/labels", ["dog", "cat", "car"])
    """
    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, json_file)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Get image filename
        image_filename = data.get("imagePath", os.path.splitext(json_file)[0] + ".jpg")

        # Get image dimensions
        img_width, img_height = data.get("imageWidth"), data.get("imageHeight")
        if not img_width or not img_height:
            print(f"⚠️ Warning: Missing image size in {json_file}, skipping...")
            continue

        # Define label file path
        label_file_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}.txt")

        with open(label_file_path, "w") as label_f:
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "polygon":
                    continue  # Ignore non-polygon annotations

                class_name = shape.get("label")
                if class_name not in class_names:
                    continue  # Skip unknown class

                class_id = class_names.index(class_name)

                # Extract and normalize polygon points
                points = shape.get("points", [])
                if len(points) < 3:
                    print(f"⚠️ Warning: Skipping invalid polygon in {json_file}")
                    continue

                polygon = [f"{x / img_width} {y / img_height}" for x, y in points]

                # Write YOLO segmentation format
                label_f.write(f"{class_id} " + " ".join(polygon) + "\n")

    print(f"✅ Conversion completed! YOLOv8 segmentation labels saved in {output_dir}")
