import cv2
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, LineString, MultiPoint, Point
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image


def get_obj_polygon(results, image, target_classes=["CEJ", "bone_level", "tooth"]):
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


def get_minor_axes(polygons):
    """
    Given a list of polygons, return a list of their minor axes as LineStrings.

    Parameters:
        polygons (list of shapely.geometry.Polygon): List of polygons.

    Returns:
        List of shapely.geometry.LineString: Minor axes of the polygons.
    """
    minor_axes = []

    for polygon in polygons:
        # Get exterior coordinates (excluding the duplicate last point)
        x, y = polygon.exterior.xy
        points = np.column_stack((x[:-1], y[:-1]))

        # Compute centroid
        centroid = np.array([polygon.centroid.x, polygon.centroid.y])

        # Apply PCA to get principal components
        pca = PCA(n_components=2)
        pca.fit(points - centroid)  # Center the points

        # Get minor axis direction using the second principal component
        minor_axis_dir = pca.components_[1]
        minor_length = pca.explained_variance_[1] ** 0.5  # Scale by variance

        # Compute minor axis line (extend it for visualization)
        minor_axis_start = centroid - minor_axis_dir * minor_length * 100
        minor_axis_end = centroid + minor_axis_dir * minor_length * 100

        # Convert to LineString
        minor_axis_line = LineString([minor_axis_start, minor_axis_end])
        minor_axes.append(minor_axis_line)

    return minor_axes


def get_single_major_axis(polygon):
    """
    Given a Shapely polygon, return its major axis as a LineString.

    Parameters:
        polygon (shapely.geometry.Polygon): The polygon for which to compute the major axis.

    Returns:
        shapely.geometry.LineString: The major axis of the polygon.
    """
    # Get exterior coordinates, excluding the duplicate last point
    x, y = polygon.exterior.xy
    points = np.column_stack((x[:-1], y[:-1]))

    # Compute centroid
    centroid = np.array([polygon.centroid.x, polygon.centroid.y])

    # Apply PCA to get principal components
    pca = PCA(n_components=2)
    pca.fit(points - centroid)  # Center the points

    # Get major axis direction and length
    major_axis_dir = pca.components_[0]
    major_length = pca.explained_variance_[0] ** 0.5

    # Extend the line for visualization purposes
    major_axis_start = centroid - major_axis_dir * major_length * 100
    major_axis_end = centroid + major_axis_dir * major_length * 100

    return LineString([major_axis_start, major_axis_end])


def get_minor_axis_from_point(polygon, given_point):
    """
    Given a Shapely polygon, return its minor axis as a LineString drawn from a given point.

    Parameters:
        polygon (shapely.geometry.Polygon): The polygon for which to compute the minor axis.
        given_point (tuple or shapely.geometry.Point): The point from which to draw the minor axis.

    Returns:
        shapely.geometry.LineString: The minor axis of the polygon, anchored at the given point.
    """
    # Get exterior coordinates, excluding the duplicate last point.
    x, y = polygon.exterior.xy
    points = np.column_stack((x[:-1], y[:-1]))

    # Compute the polygon's centroid for PCA computation.
    centroid = np.array([polygon.centroid.x, polygon.centroid.y])

    # Apply PCA on the centered points.
    pca = PCA(n_components=2)
    pca.fit(points - centroid)

    # Get the minor axis direction and its length.
    minor_axis_dir = pca.components_[1]
    minor_length = pca.explained_variance_[1] ** 0.5

    # Convert the given_point to a numpy array.
    if isinstance(given_point, Point):
        origin = np.array([given_point.x, given_point.y])
    else:
        origin = np.array(given_point)

    # Extend the line from the given point for visualization.
    extension_factor = 100  # Adjust the factor as needed.
    minor_axis_start = origin - minor_axis_dir * minor_length * extension_factor
    minor_axis_end = origin + minor_axis_dir * minor_length * extension_factor

    return LineString([minor_axis_start, minor_axis_end])


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


def calculate_ratios(d1, d2):
    """
    Calculate ratios from two lists of distances.

    Parameters:
        d1, d2 (list of float or None): Lists of distances.

    Returns:
        list: A list of ratios computed as d1_value/d2_value. If either value is None
              or d2_value is zero, returns None or 0 accordingly.
    """
    ratios = [
        (
            None
            if d1_val is None or d2_val is None
            else (d1_val / d2_val if d2_val != 0 else 0)
        )
        for d1_val, d2_val in zip(d1, d2)
    ]
    return ratios


def save_ratios_to_csv(ratios, image_filename, csv_path="./outputs/ratios.csv"):
    """
    Save the calculated ratios along with the image filename to a CSV file.

    Parameters:
        ratios (list): List of ratio values.
        image_filename (str): The filename of the image.
        csv_path (str): Path to the CSV file.
    """
    data = {"filename": [image_filename], "ratios": [ratios]}
    df = pd.DataFrame(data)
    # Append to CSV; create header only if file doesn't exist
    df.to_csv(
        csv_path, mode="a", header=not pd.io.common.file_exists(csv_path), index=False
    )


def plot_points_on_image(list1, list2, list3, image, ratios):
    """
    Plot points from three lists on an image and annotate the points from list2 with ratios.

    Parameters:
        list1, list2, list3 (list of shapely.geometry.Point or None): Lists of points.
        image (PIL.Image or np.ndarray): The background image.
        ratios (list): List of ratios to annotate near the points from list2.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Ensure image is a PIL Image
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Plot points for each list (skipping None values)
    for i, point in enumerate(list1):
        if point is not None:
            ax.plot(point.x, point.y, "bo", label=f"Point1-{i+1}")
    for i, point in enumerate(list2):
        if point is not None:
            ax.plot(point.x, point.y, "go", label=f"Point2-{i+1}")
    for i, point in enumerate(list3):
        if point is not None:
            ax.plot(point.x, point.y, "ro", label=f"Point3-{i+1}")

    # Annotate each list2 point with its corresponding ratio
    for i, (point, ratio) in enumerate(zip(list2, ratios)):
        if point is not None:
            text = f"Ratio: {ratio:.2f}" if ratio is not None else "Ratio: N/A"
            ax.text(point.x, point.y - 10, text, fontsize=12, color="red", ha="center")

    # Set plot limits based on the image size
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)  # Invert y-axis for image coordinates
    ax.axis("off")

    return fig


def plot_points_and_ratios(
    cejs,
    bones,
    roots,
    image,
    ratios,
    save_path="./outputs/plot.png",
):
    """
    Process the given points, calculate ratios from pairwise distances, plot the points with ratio annotations,
    and save both the resulting plot and the ratio data.

    Parameters:
        list1, list2, list3 (list of shapely.geometry.Point or None): Lists of points.
        image (PIL.Image or np.ndarray): The background image.
        d1, d2 (list of float or None): Lists of distances.
        save_path (str): Path to save the output image.
        csv_path (str): Path to save the CSV file.
    """
    # Calculate ratios using the helper function

    # Plot the points and annotate them with the ratios
    fig = plot_points_on_image(cejs, bones, roots, image, ratios)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def filter_small_polygons(polygons, relative_threshold=0.3):
    """
    Filters out polygons with area smaller than a specified percentage of the largest polygon's area.

    Parameters:
        polygons (list of shapely.geometry.Polygon): List of polygons to filter.
        relative_threshold (float): The fraction (between 0 and 1) of the largest polygon's area that
                                    a polygon must have to be kept. Default is 0.3 (30%).

    Returns:
        List of shapely.geometry.Polygon: Filtered polygons that have an area larger than or equal to
                                          relative_threshold * (max polygon area).
    """
    if not polygons:
        return []

    # Determine the area of the largest polygon in the list.
    max_area = max(polygon.area for polygon in polygons)
    threshold_area = relative_threshold * max_area

    # Filter out polygons that are smaller than the threshold_area.
    filtered_polygons = [
        polygon for polygon in polygons if polygon.area >= threshold_area
    ]

    return filtered_polygons


def get_lowest_coordinate(polygon):
    """
    Given a Shapely polygon, return its lowest coordinate as a Shapely Point.
    The lowest coordinate is defined as the coordinate with the minimum y-value.
    In case of ties, the coordinate with the minimum x-value is chosen.

    Parameters:
        polygon (shapely.geometry.Polygon): A Shapely Polygon object.

    Returns:
        shapely.geometry.Point: The lowest coordinate (x, y) of the polygon.
    """
    # Get the exterior coordinates of the polygon.
    coords = list(polygon.exterior.coords)
    # Determine the coordinate with the smallest y value; if there's a tie, choose the smallest x.
    lowest_coord = max(coords, key=lambda p: (p[1], p[0]))
    return Point(lowest_coord)


def get_parallel_major_axes(polygon, extension_factor=100, shift_ratio=0.4):
    """
    Given a Shapely polygon, compute its major axis using PCA, then create two lines parallel
    to the major axis, shifted along the minor axis by a fraction (shift_ratio) of the polygon's width.
    One line is shifted to one side and the other to the opposite side.
    The function returns the two lines sorted by their average x coordinate (from left to right).

    Parameters:
        polygon (shapely.geometry.Polygon): The polygon for which to compute the axes.
        extension_factor (float): Factor to extend the major axis for visualization.
        shift_ratio (float): Fraction of the polygon's width along the minor axis to shift.

    Returns:
        tuple: Two shapely.geometry.LineString objects representing the parallel major axes,
               sorted such that the first line is further left (smaller average x) than the second.
    """
    # Get the exterior coordinates (exclude the duplicate last point)
    x, y = polygon.exterior.xy
    points = np.column_stack((x[:-1], y[:-1]))

    # Compute the centroid as a numpy array
    centroid = np.array([polygon.centroid.x, polygon.centroid.y])

    # Center the points for PCA
    centered_points = points - centroid

    # Apply PCA to get principal components (axes)
    pca = PCA(n_components=2)
    pca.fit(centered_points)

    # Major axis direction (first principal component) and minor axis direction (second principal component)
    major_axis_dir = pca.components_[0]
    minor_axis_dir = pca.components_[1]

    # Calculate the polygon's width along the minor axis:
    projections = centered_points.dot(minor_axis_dir)
    width = projections.max() - projections.min()

    # Calculate the offset distance (shift_ratio of the polygon's width)
    offset = shift_ratio * width

    # Determine the new "centroids" for the two parallel lines by shifting along the minor axis
    left_centroid = centroid + minor_axis_dir * offset
    right_centroid = centroid - minor_axis_dir * offset

    # Use the square root of the explained variance along the major axis as a scale for visualization
    major_length = np.sqrt(pca.explained_variance_[0])

    # Compute endpoints for the two parallel lines along the major axis, anchored at the shifted centroids
    left_line_start = left_centroid - major_axis_dir * major_length * extension_factor
    left_line_end = left_centroid + major_axis_dir * major_length * extension_factor

    right_line_start = right_centroid - major_axis_dir * major_length * extension_factor
    right_line_end = right_centroid + major_axis_dir * major_length * extension_factor

    left_line = LineString([left_line_start, left_line_end])
    right_line = LineString([right_line_start, right_line_end])

    # Sort the lines by the average x coordinate of their endpoints (from left to right)
    lines = [left_line, right_line]
    lines.sort(key=lambda line: np.mean([pt[0] for pt in line.coords]))

    return tuple(lines)
