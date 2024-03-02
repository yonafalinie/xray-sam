################################################################################
# Example : Perform annotation check/modification/draw for coco format json
# Copyright (c) 2024 - Neelanjan Bhowmik
# License: MIT
################################################################################
import argparse
import numpy as np
import cv2
import os
import pandas as pd
import random
from pycocotools.coco import COCO
from shapely.geometry import Polygon, Point
################################################################################


def polygon_centroid(segmentation):
    x_coords = np.array(segmentation[::2])
    y_coords = np.array(segmentation[1::2])
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    return x_center, y_center
################################################################################


def find_random_points_in_polygon(segmentation, n=3):
    polygon = np.array(segmentation).reshape((-1, 2)).astype(np.float32)
    points = []
    while len(points) < n:
        rand_x = random.uniform(polygon[:, 0].min(), polygon[:, 0].max())
        rand_y = random.uniform(polygon[:, 1].min(), polygon[:, 1].max())
        if cv2.pointPolygonTest(polygon, (rand_x, rand_y), False) >= 0:
            points.append((rand_x, rand_y))
    return points
################################################################################

# # Create polygon object
# polygon = Polygon(polygon_points)
def rand_point_within_poly(polygon, num_attempts=1000):
    min_x, min_y, max_x, max_y = polygon.bounds
    for _ in range(num_attempts):
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return random_point
    return None

################################################################################

def process_annotations(coco_annotation_path, images_dir, output_dir, csv_path):
    coco = COCO(coco_annotation_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # df_centroids = pd.DataFrame(columns=['image_id', 'annotation_id', 'image_name', 'centroid_x', 'centroid_y', 'category', 'bbox_x_top_left',
    #                             'bbox_y_top_left', 'bbox_x_bottom_right', 'bbox_y_bottom_right', 'point_1_x', 'point_1_y', 'point_2_x', 'point_2_y', 'point_3_x', 'point_3_y'])

    df_centroids = pd.DataFrame(columns=['image_id', 'annotation_id', 'image_name', 'centroid_x', 'centroid_y', 'category', 'bbox_x_top_left',
                                'bbox_y_top_left', 'bbox_x_bottom_right', 'bbox_y_bottom_right', 'point_1_x', 'point_1_y', 'point_2_x', 'point_2_y', 'point_3_x', 'point_3_y', 'pt1x_poly', 'pt1y_poly', 'pt2x_poly', 'pt2y_poly'])

    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        img = cv2.imread(img_path)

        # Process segmentation if available
        if 'segmentation' in ann and isinstance(ann['segmentation'][0], list):
            for segmentation in ann['segmentation']:
                centroid_x, centroid_y = polygon_centroid(segmentation)
                centroid_x = int(centroid_x * img.shape[1] / img_info['width'])
                centroid_y = int(
                    centroid_y * img.shape[0] / img_info['height'])
                cv2.circle(img, (centroid_x, centroid_y), 5,
                           (0, 255, 0), -1)  # Draw the centroid

            # Calculate three random points within each segmentation in the annotation
            for segmentation in ann['segmentation']:
                random_points = find_random_points_in_polygon(segmentation, 3)

                for point in random_points:
                    # Draw each random point
                    cv2.circle(img, (int(point[0]), int(
                        point[1])), 5, (0, 0, 255), -1)
                    
                # Generate random points within the polygon
                # Convert coordinates into pairs
                polygon_points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]

                # Create polygon object
                polygon = Polygon(polygon_points)    

                random_point1 = rand_point_within_poly(polygon)
                random_point2 = rand_point_within_poly(polygon)

                # Extract x and y coordinates
                pt1x_poly, pt1y_poly = random_point1.x, random_point1.y
                pt2x_poly, pt2y_poly = random_point2.x, random_point2.y

            output_path = os.path.join(output_dir, img_info['file_name'])
            # cv2.imwrite(output_path, img)

        # Process bounding box
        bbox = ann['bbox']
        x_top_left, y_top_left, width, height = bbox
        x_bottom_right = x_top_left + width
        y_bottom_right = y_top_left + height

        # Get category name
        cat_id = ann['category_id']
        cat_name = coco.loadCats(cat_id)[0]['name']

        # Append new row to the DataFrame
        new_row = {
            'image_id': ann['image_id'],
            'annotation_id': ann['id'],
            'image_name': img_info['file_name'],
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'category': cat_name,
            'bbox_x_top_left': x_top_left,
            'bbox_y_top_left': y_top_left,
            'bbox_x_bottom_right': x_bottom_right,
            'bbox_y_bottom_right': y_bottom_right,
            'point_1_x': random_points[0][0], 'point_1_y': random_points[0][1],
            'point_2_x': random_points[1][0], 'point_2_y': random_points[1][1],
            'point_3_x': random_points[2][0], 'point_3_y': random_points[2][1],
            'pt1x_poly': pt1x_poly, 'pt1y_poly': pt1y_poly,
            'pt2x_poly': pt2x_poly, 'pt2y_poly': pt2y_poly
        }
        # Convert new_row dictionary to a DataFrame
        new_row_df = pd.DataFrame([new_row])
        df_centroids = pd.concat([df_centroids, new_row_df], ignore_index=True)

    # Save DataFrame to CSV
    df_centroids.to_csv(csv_path, index=False)
################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process COCO annotations to generate centroids and bounding boxes.")
    parser.add_argument("--cocogt", required=True,
                        help="Path to the COCO annotation JSON file.")
    parser.add_argument("--image", required=True,
                        help="Directory containing the images.")
    parser.add_argument("--output", required=True,
                        help="Output directory for processed images.")
    parser.add_argument("--csvpath", required=True,
                        help="Path to save the output CSV file.")
    args = parser.parse_args()

    d = os.path.dirname(args.csvpath)
    os.makedirs(d, exist_ok=True)
    
    process_annotations(args.cocogt,
                        args.image, args.output, args.csvpath)
