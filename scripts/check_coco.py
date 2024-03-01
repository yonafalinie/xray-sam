from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import cv2
import os
import numpy as np


def draw_annotation(image, annotation):
    # Draw bounding box
    bbox = annotation['bbox']
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)

    # Draw segmentation mask
    if 'segmentation' in annotation:
        rle = annotation['segmentation']
        if isinstance(rle, dict):  # RLE format
            mask = mask_utils.decode(rle)
        else:  # Polygon format
            # Create an empty mask and draw the polygons
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for seg in rle:
                poly = np.array(seg).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], 1)

        if mask is not None:
            # Convert mask to three channels and same data type as image
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            colored_mask = mask * np.array([0, 0, 255], dtype=image.dtype)
            image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

    return image


# Paths
coco_annotation_file = '/home2/projects/datasets/pidray/SAM/pidray_train__centroid.json'
images_dir = '/home2/projects/datasets/pidray/image/train/'
output_dir = '/home2/projects/datasets/pidray/SAM/browse/'

# Initialize COCO API
coco = COCO(coco_annotation_file)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image
for img_id in coco.getImgIds():
    # Load image
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(images_dir, img_info['file_name'])
    image = cv2.imread(img_path)

    # Verify if image is loaded
    if image is None:
        print(f"Failed to load image {img_info['file_name']}")
        continue

    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Draw annotations on the image
    for ann in annotations:
        image = draw_annotation(image, ann)

    # Save the annotated image
    cv2.imwrite(os.path.join(output_dir, img_info['file_name']), image)

print("Completed drawing annotations.")
