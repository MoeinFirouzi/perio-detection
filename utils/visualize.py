import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize_polygon(polygons: list, image):
    overlay_image = image.copy()

    # Loop through the polygons and draw them on the image
    for polygon in polygons:
        # Extract exterior coordinates of the polygon
        coords = np.array(polygon.exterior.coords).astype(np.int32)

        # Draw the polygon on the image (filled or outline)
        cv2.polylines(
            overlay_image, [coords], isClosed=True,
            color=(255, 0, 0), thickness=2
        )  # Red outline

    # Display the image with overlaid polygons
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_image)
    plt.axis("off")
    plt.show()
