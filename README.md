# Evaluating Performance of Segment Anything Model for Beyond Visible Spectrum Imagery

## :wrench: Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

1. Install Segment Anything:

    clone the repository locally and install with

    ~~~
    pip3 install -e .
    ~~~
2. Install the requirements
    ~~~
    pip3 install -r requirements.txt
    ~~~

## :plate_with_cutlery: Getting started

1. Run the `scripts/coco2csv` with different command line options:

    ~~~
    coco2csv.py [-h] --cocogt COCOGT --image IMAGE --output OUTPUT --csvpath CSVPATH

    Process COCO annotations to generate centroids and bounding boxes.

    options:
      -h, --help         show this help message and exit
      --cocogt COCOGT    Path to the COCO annotation JSON file.
      --image IMAGE      Directory containing the images.
      --output OUTPUT    Output directory for processed images.
      --csvpath CSVPATH  Path to save the output CSV file.
    ~~~

2. Run the `scripts/sam_prompt` with different command line options:

    ~~~
    sam_prompt.py [-h] [--image IMAGE] [--cocogt COCOGT] [--csvpath CSVPATH] [--weight WEIGHT] [--prompt_centroid]
                        [--prompt_bbox] [--prompt_rand] [--output_json OUTPUT_JSON]

    sam prompt

    options:
      -h, --help            show this help message and exit
      --image IMAGE         image file/directory path (default: None)
      --cocogt COCOGT       Coco annotation gt file path (default: None)
      --csvpath CSVPATH     csv file path (default: None)
      --weight WEIGHT       weight file path (default: weights/sam_vit_h_4b8939.pth)
      --prompt_centroid     sam prompt centroid (default: False)
      --prompt_bbox         sam prompt bbox (default: False)
      --prompt_rand         sam prompt random points (default: False)
      --output_json OUTPUT_JSON
                            Output json file path. (default: None)
    ~~~

3. Run the `scripts/evaluate` with different command line options:

    ~~~
    evaluate.py [-h] --cocogt COCOGT --detjson DETJSON --iou_type {bbox,segm} [--statcsv STATCSV]

    Evaluate COCO metrics.

    options:
      -h, --help            show this help message and exit
      --cocogt COCOGT       Path to the ground truth COCO annotation file.
      --detjson DETJSON     Path to the detection COCO annotation file.
      --iou_type {bbox,segm}
                            Type of IoU metric ('bbox' or 'segm').
      --statcsv STATCSV     output directory path to save stats file
    ~~~

4. Run the `scripts/drawmask` with different command line options:

    ~~~
    drawmask.py [-h] [--gtfile GTFILE] [--image IMAGE] [--output OUTPUT]

    options:
      -h, --help       show this help message and exit
      --gtfile GTFILE  ground truth [in coco format] json file path
      --image IMAGE    image file/directory path
      --output OUTPUT  output directory path to save images
    ~~~


## :frog: Reference
If you use this repo and like it, use this to cite it:
```tex
@misc{}
```