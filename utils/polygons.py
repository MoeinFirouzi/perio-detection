from shapely.geometry import Polygon
import cv2
import numpy as np


def get_obj_polygon(results, image, target_classes=["CEJ", "bone_level"]):
    """
    Extracts polygons for specific target classes (e.g., "CEJ", "bone_level")
    from YOLO segmentation results.

    Parameters:
        results (list): YOLO segmentation results containing masks and class
                        predictions.
        image (numpy.ndarray): Original image for resizing masks.
        target_classes (str or list): The name(s) of the class(es) to extract
                                      polygons for. Default is ["CEJ"].

    Returns:
        dict: A dictionary where keys are target classes and values are lists
              of Shapely Polygons for each class.
    """
    # Validate inputs
    if not results or not isinstance(results, list):
        raise ValueError("Results must be a non-empty list.")
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Image must be a valid numpy array.")
    if isinstance(target_classes, str):
        target_classes = [target_classes]  # Convert single class to list

    # Extract necessary data from YOLO results
    masks = results[0].masks.data.cpu().numpy()  # Segmentation masks
    classes = results[0].boxes.cls.cpu().numpy()  # Class IDs for each mask
    class_names = results[0].names  # Class name mapping

    # Get the target class IDs
    target_class_ids = {
        name: idx for idx, name in class_names.items() if name in target_classes
    }
    if not target_class_ids:
        raise ValueError(
            f"None of the target classes {target_classes} found in model classes."
        )

    # Initialize dictionary to store polygons
    polygons_by_class = {cls_name: [] for cls_name in target_classes}

    # Loop through masks and filter by target class
    for mask, class_id in zip(masks, classes):
        for target_class, target_class_id in target_class_ids.items():
            if int(class_id) == target_class_id:
                # Convert mask to binary and resize to image dimensions
                binary_mask = (mask > 0).astype(np.uint8)
                resized_mask = cv2.resize(
                    binary_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Find contours of the mask
                contours, _ = cv2.findContours(
                    resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Create polygons from contours
                for contour in contours:
                    if len(contour) >= 3:
                        polygon = Polygon(contour[:, 0, :])
                        polygons_by_class[target_class].append(polygon)

    return polygons_by_class
