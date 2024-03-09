################################################################################
# Example : Perform annotation check/modification/draw for coco format json
# Copyright (c) 2024 - Neelanjan Bhowmik
# License: MIT
################################################################################

from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import cv2
import os
import numpy as np
import argparse
import tqdm
from PIL import Image, ImageDraw

################################################################################


def draw_annotation(image, annotation, category_id, cat_name, segm):
    # Draw bounding box
    bbox = annotation['bbox']
    # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(
    #     bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(
        bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (144, 238, 144), 2)

    cat = f'{category_id}|{cat_name}'
    # cv2.putText(image,
    #             cat,
    #             (int(bbox[0]), int(bbox[1]) - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0, 0, 255),
    #             1,
    #             )
    
    # Draw segmentation mask
    if segm:
        if 'segmentation' in annotation:
            
            if 'counts' in annotation['segmentation']:
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
                    mask = mask.astype(image.dtype)
                    colored_mask = np.zeros_like(mask)
                    # colored_mask[mask == 1] = [255, 255, 255]  # White mask
                    colored_mask = mask * np.array([255, 255, 255], dtype=image.dtype)
                    alpha = 0.5  # Transparency factor
                    blended_image = cv2.addWeighted(image, 1, colored_mask, beta=0.5, gamma=0)
                    image = blended_image
                    # colored_mask = mask * np.array([0, 0, 255], dtype=image.dtype)
                    # image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

            else:
                segmentation = annotation["segmentation"]
                if len(segmentation) > 0:

                    
                    # A list of points, each point is a list containing 2 elements (x, y)
                    points = np.array(segmentation, dtype=np.int32)
                    points = np.array(points).reshape((-1, 1, 2))
                    # Create an empty mask with the same dimensions as the original image
                    # Make sure to create a mask for all three channels if your image is in color
                    mask = np.zeros_like(image)
                    
                    # Draw the polygon on the mask with white color and fill the inside
                    cv2.fillPoly(mask, [points], color=(255, 255, 255))

                    # Blend the original image with the mask
                    # Calculate the weighted sum of the input image and the mask
                    # Since we want 50% transparency, we use 0.5 for both weights and 0 for the gamma parameter
                    blended_img = cv2.addWeighted(src1=image, alpha=1, src2=mask, beta=0.5, gamma=0)

                    # Update the original image with the blended image
                    image = blended_img


                    # # Create a white mask
                    # mask = np.zeros_like(image, dtype=np.uint8)

                    # # Draw the contour on the mask
                    # cv2.drawContours(
                    #     mask,
                    #     [
                    #         np.array(segmentation)
                    #         .reshape((-1, 1, 2))
                    #         .astype(np.int32)
                    #     ],
                    #     -1,
                    #     (255, 255, 255),
                    #     thickness=cv2.FILLED,
                    # )
                    # # Blend the mask with the original image using transparency
                    # img = cv2.addWeighted(image, 1, mask, 0.5, 0)

    return image
################################################################################


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfile",
                        type=str,
                        help="ground truth [in coco format] json file path")
    parser.add_argument("--image", type=str, help="image file/directory path")
    parser.add_argument("--segm", action="store_true", help="draw segmentation mask on image")
    parser.add_argument("--output",
                        type=str,
                        default='./data',
                        help="output directory path to save images")

    args = parser.parse_args()

    # Initialize COCO API
    coco = COCO(args.gtfile)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Process each image
    for img_id in tqdm.tqdm(coco.getImgIds()):
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image, img_info['file_name'])
        image = cv2.imread(img_path)

        # imageP = Image.open(img_path)
        # # Create an identical transparent image for the mask
        # mask = Image.new('RGBA', imageP.size, (255, 255, 255, 0))

        # Verify if image is loaded
        if image is None:
            print(f"Failed to load image {img_info['file_name']}")
            continue

        # Get annotations for the image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Draw annotations on the image
        for ann in annotations:
            category_id = ann["category_id"]
            catIds = coco.loadCats(ids=[ann["category_id"]])
            cat_name = catIds[0]["name"]
            image = draw_annotation(image, ann, category_id, cat_name, args.segm)

            # segmentation = ann["segmentation"]
            # draw = ImageDraw.Draw(mask)

            # # Define the color with 50% opacity (128 out of 255 for alpha channel)
            # fill_color = (255, 255, 255, 128)  # RGBA, where A is the alpha channel

            # # Loop through your segmentation data if you have multiple polygons
            # for polygon in segmentation:
            #     # The polygon needs to be flattened for PIL's polygon method
            #     draw.polygon(polygon, outline="red", fill=fill_color)

            # # Blend the original image with the mask
            # # Since the original image doesn't have an alpha channel, we'll convert it to 'RGBA'
            # image_with_transparency = Image.alpha_composite(imageP.convert('RGBA'), mask)

            # # Convert back to RGB to save as JPG or show
            # image_with_transparency = image_with_transparency.convert("RGB")
            # image_with_transparency.save(os.path.join(args.output, img_info['file_name']))

        # Save the annotated image
        
        p = os.path.dirname(img_info['file_name'])
        if len(p) > 0:
            os.makedirs(os.path.join(args.output, p), exist_ok=True)

        cv2.imwrite(os.path.join(args.output,
                    img_info['file_name']),
                    image)

    print('\n[done]\n')
################################################################################


if __name__ == '__main__':
    main()
