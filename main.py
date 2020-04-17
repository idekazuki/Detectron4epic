import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import os
import json
import torch

from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset
from gulpio.transforms import Scale, CenterCrop, Compose, UnitNorm

from read_gulpio import EpicDataset


class_type = 'noun'
rgb_train = EpicVideoDataset('../../epic/data/processed/gulp/rgb_train', class_type)
transforms = Compose([])
dataset = EpicDataset(transforms)
segment_uids = list(rgb_train.gulp_dir.merged_meta_dict.keys())
exsample_segment = rgb_train.video_segments[10]
exsample_frames = rgb_train.load_frames(exsample_segment)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

for batch_num, (data, label) in enumerate(dataloader):
    frame = data[0].to('cpu').detach().numpy().copy()
    frame = frame.transpose(1, 2, 3, 0)
    frame = np.squeeze(frame)
    break
im = frame
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("./output.jpg", v.get_image()[:, :, ::-1])
