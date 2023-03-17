import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode
import cv2
import pycocotools.mask as mask_utils
import numpy as np
import random
from detectron2.utils.visualizer import Visualizer

from tqdm import tqdm

import torch
import ctypes

# import base64
# import zlib
# from pycocotools import _mask as coco_mask

classes = ['car', 'pedestrian']


# # https://www.kaggle.com/code/robertkag/rle-to-mask-converter
# def rleToMask(rleString,height,width):
#   rows,cols = height,width
#   rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
#   rlePairs = np.array(rleNumbers).reshape(-1,2)
#   img = np.zeros(rows*cols,dtype=np.uint8)
#   for index,length in rlePairs:
#     index -= 1
#     img[index:index+length] = 255
#   img = img.reshape(cols,rows)
#   img = img.T
#   return img

# def rle2mask(rle, img_shape):
#     mask = np.zeros(img_shape)
#     if rle is None:
#         return mask

#     rle = np.array([int(x) for x in rle.split()])
#     rle = rle.reshape(-1, 2)
#     for start, length in rle:
#         start -= 1
#         mask[start // img_shape[1]: (start + length) // img_shape[1], start % img_shape[1]: (start + length) % img_shape[1]] = 1
#         start += length
#     return mask



# def decodeToBinaryMask(rleCodedStr, imWidth, imHeight):
#     uncodedStr = base64.b64decode(rleCodedStr)
#     uncompressedStr = zlib.decompress(uncodedStr,wbits = zlib.MAX_WBITS)
#     detection ={
#         'size': [imWidth, imHeight],
#         'counts': uncompressedStr
#     }
#     detlist = []
#     detlist.append(detection)
#     mask = coco_mask.decode(detlist)
#     binaryMask = mask.astype('bool') 
#     return binaryMask

# # Chatgpt rules
# def rle2mask(rle, img_shape):
#     """
#     Convert RLE(run length encoding) string to numpy array.

#     Args:
#         rle (str): RLE string
#         img_shape (tuple): Target image shape (height, width)

#     Returns:
#         np.array: Image as numpy array.
#     """
#     # Extract values and lengths of each segment
#     rle = ''.join(filter(str.isdigit, rle))
#     values = list(map(int, rle[::2]))
#     lengths = list(map(int, rle[1::2]))

#     # Decode RLE data and create a binary mask
#     mask = np.zeros(img_shape[0] * img_shape[1], dtype=np.uint8)
#     for value, length in zip(values, lengths):
#         mask[value : value + length] = 1
#     mask = mask.reshape(img_shape).T

#     # Decompress mask if necessary
#     if mask.max() > 1:
#         mask = zlib.decompress(mask)

#     return mask


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
        "category_id": class_id,
        "bbox": bbox,
        "bbox_mode": BoxMode.XYWH_ABS,
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
    for subset in ["train", "val"]:
        DatasetCatalog.register(f"kitti_{subset}", lambda subset=subset: get_kitti_dicts(subset))
        print(f"Successfully registered 'kitti_{subset}'!")
        MetadataCatalog.get(f"kitti_{subset}").thing_classes = classes
    
    kitty_metadata = MetadataCatalog.get("kitti_train")
    return kitty_metadata



if __name__ == "__main__":

    kitty_metadata = register_kitti_dataset()
    dataset_dicts = get_kitti_dicts("train")

    count = 0 
    # for d in random.sample(dataset_dicts, 3):
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitty_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        # cv2.imshow(out.get_image()[:, :, ::-1])
        cv2.imwrite(f"test_{count}.png", out.get_image()[:, :, ::-1])
        break