import cv2
import numpy as np

from shapely.geometry import Polygon, LineString, MultiPoint, Point
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image


def get_obj_polygon(
    results, image, target_classes=["CEJ", "bone_level", "tooth"]
):
    """
    Extracts polygons for specific target classes (e.g., "CEJ", "bone_level")
    from YOLO segmentation results and returns them in left-to-right order.

    Parameters:
        results (list):
            YOLO segmentation results containing masks and class predictions.
        image (numpy.ndarray):
            Original image for resizing masks.
        target_classes (str or list):
            The name(s) of the class(es) to extract polygons for.
            Default is ["CEJ", "bone_level", "tooth"].

    Returns:
        dict:
            A dictionary where keys are target classes and values are lists
            of Shapely Polygons for each class sorted from left to right.
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

    # Sort the polygons for each class in left-to-right order
    # (using the minimum x-coordinate from polygon.bounds which returns (minx, miny, maxx, maxy))
    for target_class in polygons_by_class:
        polygons_by_class[target_class] = sorted(
            polygons_by_class[target_class], key=lambda poly: poly.bounds[0]
        )

    return polygons_by_class


def find_lower_intersection(major_axes, polygons):
    """
    Finds the lower (smallest y-coordinate) intersection point of each line with its corresponding polygon.

    Parameters:
        major_axes (list of shapely.geometry.LineString): List of major axes (lines).
        polygons (list of shapely.geometry.Polygon): List of polygons.

    Returns:
        List of the lowest intersection points (shapely.geometry.Point or None if no intersection).
    """
    lower_intersections = []

    for axis, polygon in zip(major_axes, polygons):
        # Get the polygon's border as a LineString
        polygon_border = LineString(polygon.exterior.coords)

        # Compute intersection of the line with the polygon border
        intersection = axis.intersection(polygon_border)

        # Check if we have multiple intersection points
        if isinstance(intersection, MultiPoint):
            # Get the lowest intersection (smallest y-coordinate)
            lowest_point = max(intersection.geoms, key=lambda p: p.y)
            lower_intersections.append(lowest_point)

        elif isinstance(intersection, Point):
            lower_intersections.append(intersection)  # Single intersection

        else:
            lower_intersections.append(None)  # No intersection found

    return lower_intersections


def get_major_axes(polygons):
    """
    Given a list of polygons, return a list of their major axes as LineStrings.

    Parameters:
        polygons (list of shapely.geometry.Polygon): List of polygons.

    Returns:
        List of shapely.geometry.LineString: Major axes of the polygons.
    """
    major_axes = []

    for polygon in polygons:
        # Get exterior coordinates
        x, y = polygon.exterior.xy
        points = np.column_stack(
            (x[:-1], y[:-1])
        )  # Exclude last point (duplicate of first)

        # Compute centroid
        centroid = np.array([polygon.centroid.x, polygon.centroid.y])

        # Apply PCA to get principal components
        pca = PCA(n_components=2)
        pca.fit(points - centroid)  # Center the points

        # Get major axis direction
        major_axis_dir = pca.components_[0]
        major_length = pca.explained_variance_[0] ** 0.5  # Scale by variance

        # Compute major axis line (extend it for visualization)
        major_axis_start = centroid - major_axis_dir * major_length * 100
        major_axis_end = centroid + major_axis_dir * major_length * 100

        # Convert to LineString
        major_axis_line = LineString([major_axis_start, major_axis_end])
        major_axes.append(major_axis_line)

    return major_axes


def find_lower_intersections_for_shape(polygon, axes):
    """
    Finds the lower (higher y-coordinate) intersection points of multiple axes with a single polygon.

    Parameters:
        polygon (shapely.geometry.Polygon): The shape (polygon).
        axes (list of shapely.geometry.LineString): List of major axes (lines).

    Returns:
        List of the lowest intersection points (shapely.geometry.Point or None if no intersection).
    """
    lower_intersections = []

    # Get the polygon's border as a LineString
    polygon_border = LineString(polygon.exterior.coords)

    for axis in axes:
        # Compute intersection of the line with the polygon border
        intersection = axis.intersection(polygon_border)

        # Check if we have multiple intersection points
        if isinstance(intersection, MultiPoint):
            # Get the lower intersection (largest y-coordinate)
            lowest_point = max(intersection.geoms, key=lambda p: p.y)
            lower_intersections.append(lowest_point)

        elif isinstance(intersection, Point):
            lower_intersections.append(intersection)  # Single intersection

        else:
            lower_intersections.append(None)  # No intersection found

    return lower_intersections


def find_higher_intersections_for_shape(polygon, axes):
    """
    Finds the lower (higher y-coordinate) intersection points of multiple axes with a single polygon.

    Parameters:
        polygon (shapely.geometry.Polygon): The shape (polygon).
        axes (list of shapely.geometry.LineString): List of major axes (lines).

    Returns:
        List of the lowest intersection points (shapely.geometry.Point or None if no intersection).
    """
    lower_intersections = []

    # Get the polygon's border as a LineString
    polygon_border = LineString(polygon.exterior.coords)

    for axis in axes:
        # Compute intersection of the line with the polygon border
        intersection = axis.intersection(polygon_border)

        # Check if we have multiple intersection points
        if isinstance(intersection, MultiPoint):
            # Get the lower intersection (largest y-coordinate)
            lowest_point = min(intersection.geoms, key=lambda p: p.y)
            lower_intersections.append(lowest_point)

        elif isinstance(intersection, Point):
            lower_intersections.append(intersection)  # Single intersection

        else:
            lower_intersections.append(None)  # No intersection found

    return lower_intersections


def calculate_pairwise_distances(list1, list2):
    """
    Calculates the Euclidean distance between corresponding points in two lists.

    Parameters:
        list1 (list of shapely.geometry.Point or None): First list of points.
        list2 (list of shapely.geometry.Point or None): Second list of points.

    Returns:
        List of distances (float values) or None for any pair where one or both points are None.
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    distances = [
        (
            None
            if (p1 is None or p2 is None)
            else np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        )
        for p1, p2 in zip(list1, list2)
    ]

    return distances


def plot_points_and_ratios(list1, list2, list3, image, d1, d2):
    """
    Plots the points from three lists on an image and displays the ratio of pairwise distances (d1/d2).

    Parameters:
        list1, list2, list3 (list of shapely.geometry.Point): Lists of points.
        image (PIL.Image or np.ndarray): The background image.
        d1, d2 (list of float): Pairwise distances calculated for list1/list2 and list2/list3.
    """
    # Ensure image is a PIL image
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)  # Convert NumPy array to PIL image
    else:
        img = image  # Use as-is if already a PIL image

    # Calculate the ratios d1/d2
    ratios = [
        d1_value / d2_value if d2_value != 0 else 0
        for d1_value, d2_value in zip(d1, d2)
    ]

    # Plot the image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Plot the points from list1, list2, list3
    for i, point in enumerate(list1):
        ax.plot(point.x, point.y, "bo", label=f"Point1-{i+1}")

    for i, point in enumerate(list2):
        ax.plot(point.x, point.y, "go", label=f"Point2-{i+1}")

    for i, point in enumerate(list3):
        ax.plot(point.x, point.y, "ro", label=f"Point3-{i+1}")

    # Annotate distances and ratios between points
    for i, (d1_value, d2_value, ratio) in enumerate(zip(d1, d2, ratios)):
        # ax.text(list1[i].x, list1[i].y, f"D1: {d1_value:.2f}", fontsize=12, color='blue', ha='right')
        # ax.text(list2[i].x, list2[i].y, f"D2: {d2_value:.2f}", fontsize=12, color='green', ha='right')
        ax.text(
            list2[i].x,
            list2[i].y - 10,
            f"Ratio: {ratio:.2f}",
            fontsize=12,
            color="red",
            ha="center",
        )

    # ax.legend()
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)  # Invert y-axis to align with image coordinates
    ax.axis(False)
    plt.show()


def process_and_plot(list1, list2, list3, image):
    """
    Process the three point lists, calculate distances and ratios, and plot them on an image.
    """
    # Calculate pairwise distances between the lists
    d1 = calculate_pairwise_distances(list1, list2)
    d2 = calculate_pairwise_distances(list2, list3)

    # Plot the points and distances on the image
    plot_points_and_ratios(list1, list2, list3, image, d1, d2)
