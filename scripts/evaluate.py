
################################################################################
# Example : Perform annotation check/modification/draw for coco format json
# Copyright (c) 2024 - Neelanjan Bhowmik
# License: MIT
################################################################################
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io
from contextlib import redirect_stdout
from tabulate import tabulate
################################################################################


def csv_write(out_csv_filename,
              summary,
              summary_all):

    with open(out_csv_filename, 'w') as csv_stat:

        # csv_stat.write('|===============================|\n')
        csv_stat.write('|====coco evaluation summary====|\n')

        csv_stat.write('|**summary: all**|\n')
        for i, s in enumerate(summary):
            for j, val in enumerate(s):
                csv_stat.write(f'{val}')
                if j != len(s)-1:
                    csv_stat.write(',')
            csv_stat.write('\n')
        csv_stat.write('|===============================|\n\n')

        for string in summary_all:
            csv_stat.write(string + "\n")

################################################################################


def evaluate_classwise(gt_ann_file, dt_ann_file, iou_type):
    """
    Evaluates class-wise COCO metrics for ground truth and detection annotation files.

    Parameters:
    - gt_ann_file: Path to the ground truth COCO annotation file.
    - dt_ann_file: Path to the detection COCO annotation file.
    - iou_type: Type of IoU metric to evaluate ('bbox' for bounding box, 'segm' for segmentation).
    """

    summary = []
    summary_all = []
    ap_50 = []
    ap_95 = []
    ap_75 = []
    # Load ground truth and detection annotations
    coco_gt = COCO(gt_ann_file)
    coco_dt = COCO(dt_ann_file)  # Load detection results

    # Get category IDs and their names
    cat_ids = coco_gt.getCatIds()
    cat_names = [cat['name'] for cat in coco_gt.loadCats(cat_ids)]

    # Evaluate each category

    # Initialize COCOeval object for bounding box evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    print(f'\n|____Overall')
    coco_eval.summarize()

    f = io.StringIO()
    with redirect_stdout(f):
        print(f'|____Overall')
        coco_eval.summarize()
    out = f.getvalue()
    summary_all.append(out)
    out = out.split('\n')
    map_95 = out[1].split('=')[-1]
    map_50 = out[2].split('=')[-1]
    map_75 = out[3].split('=')[-1]

    print('\n\n|__Class-wise coco evaluation')
    for idx, cat_id in enumerate(cat_ids):
        # Evaluate on a single category
        coco_eval.params.catIds = [cat_id]

        # Print class-specific results

        print(f"\n|____Category: {cat_names[idx]} | ID: {cat_id}")
        # Run evaluation
        f = io.StringIO()
        with redirect_stdout(f):
            coco_eval.evaluate()
            coco_eval.accumulate()
        coco_eval.summarize()

        f = io.StringIO()
        with redirect_stdout(f):
            print(f'\n|____Category: {cat_id} : {cat_names[idx]}')
            coco_eval.summarize()

        out = f.getvalue()
        summary_all.append(out)
        out = out.split('\n')
        ap_95.append((out[2].split('=')[-1]))
        ap_50.append((out[3].split('=')[-1]))
        ap_75.append((out[4].split('=')[-1]))

    summary.append(['IoU'] + cat_names + ['mAP'])
    summary.append(['IoU=0.50:0.95'] + ap_95 + [map_95])
    summary.append(['IoU=0.50'] + ap_50 + [map_50])
    summary.append(['IoU=0.75'] + ap_75 + [map_75])
    return summary, summary_all

################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate COCO metrics.")
    parser.add_argument("--cocogt",
                        type=str,
                        required=True,
                        help="Path to the ground truth COCO annotation file.")
    parser.add_argument("--detjson",
                        type=str,
                        required=True,
                        help="Path to the detection COCO annotation file.")
    parser.add_argument("--iou_type",
                        type=str,
                        choices=['bbox', 'segm'],
                        required=True,
                        help="Type of IoU metric ('bbox' or 'segm').")
    parser.add_argument("--statcsv",
                        type=str,
                        default='./statistics',
                        help="output directory path to save stats file")
    args = parser.parse_args()

    summary, summary_all = evaluate_classwise(
        args.cocogt, args.detjson, args.iou_type)

    print('\n\n|====coco evaluation summary====|\n')
    print('\n|**summary: all**|\n')
    print(tabulate(summary,
                   headers="firstrow",
                   tablefmt="psql"))

    csv_write(args.statcsv,
              summary,
              summary_all)

    print('\n[done]\n')
