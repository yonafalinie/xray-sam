# Evaluating Performance of Segment Anything Model for Beyond Visible Spectrum Imagery

## :wrench: Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

## :plate_with_cutlery: Getting started

Run the `sam_prompt` with different command line options:

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

## :frog: Reference
If you use this repo and like it, use this to cite it:
```tex
@misc{}
```