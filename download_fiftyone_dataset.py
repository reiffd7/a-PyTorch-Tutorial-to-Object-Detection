import fiftyone as fo
import json
import fire
from utils import *

CLASS_MAP = {
    "person": 1,
    "vehicle": 2,
}


def download_fiftyone_dataset(dataset_name, split):
    dataset = fo.load_dataset(dataset_name)
    dataset.download_media()
    dataset.compute_metadata()
    all_objects = []
    image_paths = []

    for sample in dataset.iter_samples(progress=True):
        objects = {"boxes": [], "labels": [], "difficulties": []}
        height = sample["metadata"]["height"]
        width = sample["metadata"]["width"]
        for detection in sample["ground_truth"].detections:
            if detection["label"] in labels:
                box = [
                    int(detection["bounding_box"][0] * width),  # x0
                    int(detection["bounding_box"][1] * height),  # y0
                    int(
                        (detection["bounding_box"][0] + detection["bounding_box"][2])
                        * width
                    ),  # x1
                    int(
                        (detection["bounding_box"][1] + detection["bounding_box"][3])
                        * height  # y1
                    ),
                ]
                objects["boxes"].append(box)
                objects["labels"].append(label_map[detection["label"]])
                objects["difficulties"].append(0)
        all_objects.append(objects)
        image_paths.append(sample.local_path)

    print(all_objects)

    images_file_path = f"./Fiftyone_{split.upper()}_images.json"
    with open(images_file_path, "w") as f:
        json.dump(image_paths, f, indent=4)

    objects_file_path = f"./Fiftyone_{split.upper()}_objects.json"
    with open(objects_file_path, "w") as f:
        json.dump(all_objects, f, indent=4)


if __name__ == "__main__":
    download_fiftyone_dataset("voc-2012-val-person-car", "test")
