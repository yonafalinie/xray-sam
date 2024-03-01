################################################################################
# Example : Perform annotation check/modification/draw for coco format json
# Copyright (c) 2024 - Neelanjan Bhowmik
# License: MIT
################################################################################

import numpy as np
import torch
import cv2
import os
import sys
import json
import argparse
from tabulate import tabulate
import tqdm
import pandas as pd
from pycocotools import mask as maskUtils
from segment_anything import sam_model_registry, SamPredictor

################################################################################


# Function to convert a binary mask to COCO RLE
def mask_to_coco_rle(mask):
    rle = maskUtils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


################################################################################


def load_coco_data(file_path):
    """
    Load COCO data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded COCO data.

    """
    with open(file_path) as f:
        return json.load(f)
################################################################################


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="sam prompt"
    )
    parser.add_argument("--image", type=str, help="image file/directory path")
    parser.add_argument("--cocogt", type=str,
                        help="Coco annotation gt file path")
    parser.add_argument("--csvpath", type=str, help="csv file path")
    parser.add_argument(
        "--weight",
        type=str,
        default="weights/sam_vit_h_4b8939.pth",
        help="weight file path",
    )
    parser.add_argument(
        "--prompt_centroid", action="store_true", help="sam prompt centroid"
    )
    parser.add_argument("--prompt_bbox", action="store_true",
                        help="sam prompt bbox")
    parser.add_argument(
        "--prompt_rand", action="store_true", help="sam prompt random points"
    )
    parser.add_argument("--output_json", type=str,
                        help="Output json file path.")
    args = parser.parse_args()
    return args


################################################################################


def main():
    args = parse_args()
    t_val = []
    for arg in vars(args):
        t_val.append([arg, getattr(args, arg)])
    print(tabulate(t_val, ["input", "value"], tablefmt="psql"))

    cocodata = load_coco_data(args.cocogt)

    # category_mapping
    category_mapping = {}
    for category in cocodata["categories"]:
        category_mapping[category["name"]] = category["id"]

    # print(category_mapping)
    # Load your CSV data
    df = pd.read_csv(args.csvpath)

    # Loading model
    if args.weight:
        model_type = "vit_h"
        device = (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        sam = sam_model_registry[model_type](checkpoint=args.weight)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    else:
        print("Model weight missing!")
        exit()

    image_id_set = set()
    images = []
    annotations = []
    total_rows = df.shape[0]
    for index, row in df.iterrows():
        print(f"|__Processing: {index+1}/{total_rows}")
        image_path = os.path.join(args.image, row["image_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.prompt_centroid:
            input_point = np.array([[row["centroid_x"], row["centroid_y"]]])
            input_label = np.array(
                [1]
            )  # Assuming label 1 for all points for simplicity

        if args.prompt_rand:
            input_point = np.array(
                [
                    [row["point_1_x"], row["point_1_y"]],
                    [row["point_2_x"], row["point_2_y"]],
                ]
            )
            input_label = np.array(
                [1, 1]
            )  # Assuming label 1 for all points for simplicity

        if args.prompt_bbox:
            input_box = np.array(
                [
                    row["bbox_x_top_left"],
                    row["bbox_y_top_left"],
                    row["bbox_x_bottom_right"],
                    row["bbox_y_bottom_right"],
                ]
            )

        # Create a predictor
        predictor.set_image(image)

        if args.prompt_centroid or args.prompt_rand:

            # Predict the mask for the current image
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        if args.prompt_bbox:
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

        height, width, _ = image.shape
        if row["image_id"] not in image_id_set:
            images.append(
                {
                    "id": row["image_id"],
                    "file_name": row["image_name"],
                    "height": height,
                    "width": width,
                }
            )
            image_id_set.add(row["image_id"])

        category_id = category_mapping.get(row["category"])
        
        for mask in masks:
            rle = mask_to_coco_rle(mask.astype(np.uint8))
            bbox = maskUtils.toBbox(rle).tolist()
            area = float(maskUtils.area(rle))
            annotations.append(
                {
                    "id": row["annotation_id"],
                    "image_id": row["image_id"],
                    "category_id": category_id,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "score": 1,
                }
            )

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": cocodata["categories"],
    }

    if args.output_json:
        dirname = os.path.dirname(args.output_json)
        os.makedirs(dirname, exist_ok=True)
        # filename = os.path.basename(args.output_json)

        with open(args.output_json, "w") as f:
            json.dump(coco_data, f, indent=4)

    print("\n\n[Done]\n\n")


################################################################################

if __name__ == "__main__":
    main()
