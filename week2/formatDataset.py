import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode
import cv2
import pycocotools.mask as mask_utils
import numpy as np
import random
from detectron2.utils.visualizer import Visualizer

from tqdm import tqdm




def line_to_object(line):
    line = line.replace("\n", "").split(" ")

    #Each line of an annotation txt file is structured like this (where rle means run-length encoding from COCO): time_frame id class_id img_height img_width rle

    time_frame, obj_id, class_id, img_height, img_width, rle_aux = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]), line[5]

    if class_id > 2:
        return None

    #obj_instance_id = obj_id % 1000
    rle = {'size': [img_height, img_width], 'counts': rle_aux}
    mask = mask_utils.decode(rle)
    # mask = rle2mask(rle, (img_height, img_width))
    # if all pixels are 0, then the mask is invalid
    if mask.sum() == 0:
        return None
    y, x = np.where(mask == 1)
    bbox = [int(np.min(x)), int(np.min(y)), int(np.max(x) - np.min(x)), int(np.max(y) - np.min(y))]


    return {
        "bbox": bbox,
        "bbox_mode": BoxMode.XYWH_ABS,
        "segmentation": rle,
        "category_id": class_id,
    }

def get_kitti_dicts(subset):

    anotations_dir = "/home/group03/mcv/datasets/KITTI-MOTS/instances_txt/"
    images = "/home/group03/mcv/datasets/KITTI-MOTS/training/image_02/"

    if subset == "train":
        # sequences_id = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
        sequences_id = ["0000"]

    elif subset == "val":
        # sequences_id = ["0002", "0006", "0007", "0008","0010","0013","0014","0016","0018"]
        sequences_id = ["0002"]
        

    dataset_dicts = []
    idx = 1

    for seq_id in tqdm(sequences_id):
        sequence_txt = os.path.join(anotations_dir, seq_id+".txt")

        with open(sequence_txt, "r") as f:
            lines = f.readlines()

        # for each line in the txt file we have an integer on first position that represents the frame
        # construct a list of list named frames that contains the number of line that corresponds at each frame, for example: 
        #  frame = [[0,1],[2,3,4]]
        #  frame[0] = [0,1] -> lines 0 and 1 correspond to frame 0
        frames = []
        for i in range(len(lines)):
            if i == 0:
                frames.append([i])
            else:
                if int(lines[i].split(" ")[0]) == int(lines[i-1].split(" ")[0]):
                    frames[-1].append(i)
                else:
                    frames.append([i])


        for i, frame in enumerate(frames):
            record = {}
            time_frame = str(i).zfill(6)
            # time_frame = str(line.split(" ")[0]).zfill(6)

            filename = os.path.join(images, seq_id, time_frame + ".png")

            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            for line in frame:
                obj = line_to_object(lines[line])
                if obj is not None:
                    objs.append(obj)

            record["annotations"] = objs

            dataset_dicts.append(record)

            idx += 1
        

    return dataset_dicts


def register_kitti_dataset():
    classes = [None, 'car', 'pedestrian']
    for subset in ["train", "val"]:
        DatasetCatalog.register(f"kitti_{subset}", lambda subset=subset: get_kitti_dicts(subset))
        print(f"Successfully registered 'kitti_{subset}'!")
        MetadataCatalog.get(f"kitti_{subset}").set(thing_classes = classes)
    
    kitty_metadata = MetadataCatalog.get("kitti_train")
    return kitty_metadata



if __name__ == "__main__":

    kitty_metadata = register_kitti_dataset()
    dataset_dicts = get_kitti_dicts("train")

    for i, d in enumerate(dataset_dicts):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitty_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        name = d["file_name"].split("/")[-1].split(".")[0]
        cv2.imwrite(f"/ghome/group03/M5-Project/week2/Results/preprocessing/train_{name}.png", out.get_image()[:, :, ::-1])
        if i == 4:
            break


    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitty_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        name = d["file_name"].split("/")[-1].split(".")[0]
        cv2.imwrite(f"/ghome/group03/M5-Project/week2/Results/preprocessing/train_{name}.png", out.get_image()[:, :, ::-1])
