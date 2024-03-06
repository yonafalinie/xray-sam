import json

def convert_to_prediction_format(coco_json_path, output_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    predictions = []

    for annotation in coco_data['annotations']:
        prediction = {
            'image_id': annotation['image_id'],
            'category_id': annotation['category_id'],
            'bbox': annotation['bbox'],
            'segmentation': annotation['segmentation'],
            'score': annotation.get('score', 1.0)  # Assuming a default score of 1.0 if not provided
        }
        predictions.append(prediction)

    # prediction_data = {
    #     'annotations': predictions
    # }

    with open(output_json_path, 'w') as f:
        json.dump(predictions, f)

convert_to_prediction_format('/media/neel/hdd2tb/hdd2tb/data/projects/dataset/dbf6/SAM/dbf6_train__rand.json', 
                             '/media/neel/hdd2tb/hdd2tb/data/projects/dataset/dbf6/SAM/dbf6_train__rand_pred.json')
