import os
import argparse

import numpy as np

"""
Evalution script:

example execution:
    python eval_awa.py --gt test_images.txt --pred example_submission.txt

see example_submission.txt for correct submission format

"""


def read_animal_file(fname):
    image_label_dict = {}
    with open(fname) as f:
        for line in f:
            image, label = line.split()
            image_label_dict[image] = label

    return image_label_dict



parser = argparse.ArgumentParser()
parser.add_argument('--gt', help="ground truth labels")
parser.add_argument('--pred', help="file of predictions")
args = parser.parse_args()


gt_dict = read_animal_file(args.gt)
pred_dict = read_animal_file(args.pred)

per_class_accuracy = {"all": []}

for image in gt_dict:
    if image not in pred_dict:
        print("Error: {} not in prediction file".format(image))
        raise Exception()

    gt_label = gt_dict[image]
    pred_label = pred_dict[image]

    if gt_label == pred_label:
        per_class_accuracy["all"].append(1)
    else:
        per_class_accuracy["all"].append(0)



print("Final Accuracy: {:.3f}".format(np.mean(per_class_accuracy["all"])))
