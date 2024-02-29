import argparse
import numpy as np
import cv2
import os
import pandas as pd
import random
from pycocotools.coco import COCO

def polygon_centroid(segmentation):
    x_coords = np.array(segmentation[::2])
    y_coords = np.array(segmentation[1::2])
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    return x_center, y_center

def find_random_points_in_polygon(segmentation, n=3):
    polygon = np.array(segmentation).reshape((-1, 2)).astype(np.float32)
    points = []
    while len(points) < n:
        rand_x = random.uniform(polygon[:, 0].min(), polygon[:, 0].max())
        rand_y = random.uniform(polygon[:, 1].min(), polygon[:, 1].max())
        if cv2.pointPolygonTest(polygon, (rand_x, rand_y), False) >= 0:
            points.append((rand_x, rand_y))
    return points

def process_annotations(coco_annotation_path, images_dir, output_dir, csv_path):
    coco = COCO(coco_annotation_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_centroids = pd.DataFrame(columns=['image_id', 'annotation_id', 'image_name', 'centroid_x', 'centroid_y', 'category', 'bbox_x_top_left', 'bbox_y_top_left', 'bbox_x_bottom_right', 'bbox_y_bottom_right', 'point_1_x', 'point_1_y', 'point_2_x', 'point_2_y', 'point_3_x', 'point_3_y'])
    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        
        if 'segmentation' in ann and isinstance(ann['segmentation'][0], list):  # Process segmentation if available
            for segmentation in ann['segmentation']:
                centroid_x, centroid_y = polygon_centroid(segmentation)
                centroid_x = int(centroid_x * img.shape[1] / img_info['width'])
                centroid_y = int(centroid_y * img.shape[0] / img_info['height'])
                cv2.circle(img, (centroid_x, centroid_y), 5, (0, 255, 0), -1)  # Draw the centroid
            
            #Calculate three random points within each segmentation in the annotation
            for segmentation in ann['segmentation']:
                random_points = find_random_points_in_polygon(segmentation, 3)

                for point in random_points:
                    # Draw each random point
                    cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            
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
            'point_3_x': random_points[2][0], 'point_3_y': random_points[2][1]        
        }
        new_row_df = pd.DataFrame([new_row])  # Convert new_row dictionary to a DataFrame
        df_centroids = pd.concat([df_centroids, new_row_df], ignore_index=True)
        
    # Save DataFrame to CSV
    df_centroids.to_csv(csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process COCO annotations to generate centroids and bounding boxes.")
    parser.add_argument("--coco_annotation_path", required=True, help="Path to the COCO annotation JSON file.")
    parser.add_argument("--images_dir", required=True, help="Directory containing the images.")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed images.")
    parser.add_argument("--csv_path", required=True, help="Path to save the output CSV file.")
    args = parser.parse_args()

    process_annotations(args.coco_annotation_path, args.images_dir, args.output_dir, args.csv_path)
