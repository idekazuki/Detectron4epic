import os
import numpy as np
import pandas as pd
import itertools
import cv2
import random

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import epic_reader

def get_epic_dicts(img_dir):
    """
    structure dict for detectron2 training.
    root: /export/data/dataset/EPIC-KITCHENS/annotations-master/EPIC_train_object_labels.csv
    path: EPIC_train_object_labels.csv (bbox data)
          (label data)


    """
    json_file = os.path.join(img_dir, 'EPIC_object_remake_test.csv')
    imgs_anns = pd.read_csv(json_file, index_col=0)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.itertuples()):
        record = {}
        #if bbox data has empty, please run brew code.
       # if len(v[6]) <  6:
       #     continue

       #bbox data was str style, not tuple style, so fix it
        bbox = v[6].strip('[]()')
        bbox = bbox.replace(' ', '')
        bbox = bbox.split(',')
        bbox = [s.strip(')(') for s in bbox]
        bbox = [int(s) for s in bbox]
        bbox = bbox[:4]
        #Since the scale of csv dasta and the actual image are different, match them.
        asp = [0.2375, 0.23703703703703705, 0.2375, 0.23703703703703705]
        bbox = [int(a * b) for a, b in zip(asp, bbox)]
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        
        file_name = [v[3], v[4], str(v[5])]
        video_id = v[4]
        frame_id = v[5]
        height, width = 256, 456#cv2.imread(filename).shape[:2]
        record['file_name'] = file_name
        record['video_id'] = video_id
        record['frame_id'] = frame_id
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width
        objs = []
        obj = {
            'bbox': bbox,
            'bbox_mode': BoxMode.XYWH_ABS,
            'category_id': v[1]
        }
        objs.append(obj)
        record['annotations'] = objs

        dataset_dicts.append(record)
    return dataset_dicts

#extract uid from object csv data.
def get_frame(id_dict, video_id, frame):
    idx = id_dict[(id_dict["video_id"]==video_id) & (id_dict["stop_frame"]>=frame) & (id_dict["start_frame"]<=frame)]
    frame_id = frame - idx['start_frame']
    frame_id = frame - idx.at[idx.index[0],'start_frame']
    uid = idx.at[idx.index[0], 'uid']
    return [uid, frame_id]


if __name__ == '__main__':
    data_loader = epic_reader.EpicDataset_detectron2()
    noun_labels_path = './input_csv/EPIC_noun_classes.csv'
    noun_labels = pd.read_csv(noun_labels_path)
    noun = noun_labels.loc[:,['class_key']].values.tolist()
    noun = list(itertools.chain.from_iterable(noun))
    DatasetCatalog.register("epic_train", lambda d=0: get_epic_dicts('./input_csv'))
    MetadataCatalog.get('epic_train').set(thing_classes=noun)
    epic_metadata = MetadataCatalog.get("epic_train")

    dataset_dicts = get_epic_dicts('./input_csv')
    #Find the corresponding frame_id and uid from vido_id and frame.
    id_dict = pd.read_csv('./input_csv/EPIC_train_action_labels.csv')
    data_loader = epic_reader.EpicDataset_detectron2()
    

    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        video_id = d['video_id']
        frame = d['frame_id']
        print(video_id, frame)
        value = get_frame(id_dict, video_id, frame)
        print(value)
        uid, frame_id = value[0], value[1]
        img = data_loader[uid, frame_id]
        print(img.shape)
        visualizer = Visualizer(img[:, :, ::-1], metadata=epic_metadata, scale=0.5)
        print(d)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite('./output_img/output{}.jpg'.format(i), vis.get_image()[:, :, ::-1])

