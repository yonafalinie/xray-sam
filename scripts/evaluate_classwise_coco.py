import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_classwise(gt_ann_file, dt_ann_file, iou_type):
    """
    Evaluates class-wise COCO metrics for ground truth and detection annotation files.

    Parameters:
    - gt_ann_file: Path to the ground truth COCO annotation file.
    - dt_ann_file: Path to the detection COCO annotation file.
    - iou_type: Type of IoU metric to evaluate ('bbox' for bounding box, 'segm' for segmentation).
    """
    # Load ground truth and detection annotations
    coco_gt = COCO(gt_ann_file)
    coco_dt = COCO(dt_ann_file) # Load detection results

    # Get category IDs and their names
    cat_ids = coco_gt.getCatIds()
    cat_names = [cat['name'] for cat in coco_gt.loadCats(cat_ids)]

    # Evaluate each category
    for idx, cat_id in enumerate(cat_ids):
        # Initialize COCOeval object for bounding box evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)

        # Evaluate on a single category
        coco_eval.params.catIds = [cat_id]

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Print class-specific results
        print(f"Results for category '{cat_names[idx]}' (ID: {cat_id})")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate class-wise COCO metrics.")
    parser.add_argument("--gt_ann_file", required=True, help="Path to the ground truth COCO annotation file.")
    parser.add_argument("--dt_ann_file", required=True, help="Path to the detection COCO annotation file.")
    parser.add_argument("--iou_type", choices=['bbox', 'segm'], required=True, help="Type of IoU metric ('bbox' or 'segm').")
    args = parser.parse_args()

    evaluate_classwise(args.gt_ann_file, args.dt_ann_file, args.iou_type)
