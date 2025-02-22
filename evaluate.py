import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import argparse

from utils.polygons import (
    find_higher_intersections_for_shape,
    find_lower_intersection,
    find_lower_intersections_for_shape,
    get_major_axes,
    get_obj_polygon,
    process_and_plot,
    filter_small_polygons,
)


def predict(image_dir: str, output_dir: str, tooth_detection_model: YOLO, cej_bone_detection_model: YOLO) -> None:
    """
    Process all images in a directory, apply YOLO detection models, filter and process detected polygons,
    and save the resulting plots.

    Parameters:
        image_dir (str): Path to the directory containing images.
        output_dir (str): Path to the directory where output images will be saved.
        tooth_detection_model (YOLO): YOLO model for tooth detection.
        cej_bone_detection_model (YOLO): YOLO model for CEJ and bone level detection.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Found {len(image_files)} images to process.")

    for image_path in tqdm(image_files, desc="Processing Images", unit="image"):
        full_image_path = os.path.join(image_dir, image_path)
        filename = os.path.basename(image_path)

        # Read and convert the image
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Warning: Unable to read image {full_image_path}. Skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Run YOLO models for detection
            tooth_results = tooth_detection_model.predict(
                full_image_path, conf=0.5, verbose=False
            )
            cej_bone_results = cej_bone_detection_model.predict(
                full_image_path, conf=0.01, max_det=2, iou=0.5, verbose=False
            )

            # Extract polygons from results
            p1 = get_obj_polygon(tooth_results, image)
            p2 = get_obj_polygon(cej_bone_results, image)

            # Filter polygons based on relative area (at least 30% of the largest detected shape)
            tooth_polygons = filter_small_polygons(p1["tooth"], relative_threshold=0.3)
            cej_polygons = filter_small_polygons(p2["CEJ"], relative_threshold=0.3)
            bone_polygons = filter_small_polygons(p2["bone_level"], relative_threshold=0.3)

            # Ensure we have detected polygons before processing further
            if not tooth_polygons or not cej_polygons or not bone_polygons:
                print(f"Warning: Insufficient detected polygons for image {filename}. Skipping.")
                continue

            # Calculate major axes and intersections
            major_axes = get_major_axes(tooth_polygons)
            roots = find_lower_intersection(major_axes=major_axes, polygons=tooth_polygons)
            cejs = find_lower_intersections_for_shape(axes=major_axes, polygon=cej_polygons[-1])
            bone_levels = find_higher_intersections_for_shape(axes=major_axes, polygon=bone_polygons[-1])

            # Save the plotted output
            save_path = os.path.join(output_dir, filename)
            process_and_plot(cejs, bone_levels, roots, image, save_path=save_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print("Processing complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Process dental images to detect tooth and bone features using YOLO models."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the input image directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where processed images will be saved.",
    )
    parser.add_argument(
        "--tooth_model",
        type=str,
        required=True,
        help="Path to the YOLO model weights for tooth detection.",
    )
    parser.add_argument(
        "--cej_bone_model",
        type=str,
        required=True,
        help="Path to the YOLO model weights for CEJ and bone level detection.",
    )

    args = parser.parse_args()

    # Load the YOLO models using the provided weights
    tooth_detection_model = YOLO(args.tooth_model)
    cej_bone_detection_model = YOLO(args.cej_bone_model)

    # Run the evaluation process
    predict(args.image_dir, args.output_dir, tooth_detection_model, cej_bone_detection_model)


if __name__ == "__main__":
    main()


# example
# python evaluate.py --image_dir ./dataset/cej_bone/images/test --output_dir ./outputs --tooth_model ./models/Tooth/best.pt --cej_bone_model ./models/CEJ_Bone/best.pt
