import numpy as np
import torch
import pandas as pd

from gulpio import GulpDirectory


class EpicDataset_detectron2(torch.utils.data.Dataset):
    """
    Epic-kitchen video dataset loader.
    Construct the Epic-kitchen video dataset loader. For training and validation,
    video clip is randomly sampled from every video with random cropping, scaling, and flipping.

    """
    def __init__(self, path='../../epic/data/processed/gulp/rgb_train/', frame_size=1,
            class_type='noun'):
        """
        Construct the Epic-kitchen video dataset loader, 
        Args:
            transform (transform): How to transform video
            path (str): video path for epic dataset in gulp format.
            frame_size (int): Number of frames to retrieve from the video segment.
            class_type (str): Options includes 'noun', 'verb', 'noun+verb'.
        """
        self.frame_size = frame_size
        self.path = path
        self.class_type = class_type
        self.dataset = GulpDirectory(path)
        self.gdict = list(self.dataset.merged_meta_dict.keys())
        self.datalen = len(self.gdict)

    def __getitem__(self, val):
        """
        Given the video index, return the list of frames, label.
        Args:
            idx (int): the video index probided by the pytorch sampler.
        Returns:
            img_group (list): the frames of sampled from the video. The dimention is 
                'channel' x 'num_frames' x 'height' x 'width'.
            label (int): the label of the current video.
        """
        uid, frame_id = val
        frames, meta = self.dataset[str(uid)]

        #dist_img = meta['num_frames'] / float(self.frame_size)
        #seg_list = np.array([int(dist_img / 2.0 + dist_img * x) for x in range(self.frame_size)])
        #img_group = []
        #for j in seg_list:
        #    frame = frames[j]
        #    img_group.append(frame)
        #img_group = np.array(img_group)
        #img_group = img_group.transpose(3, 0, 1, 2)
        #label = meta['noun_class']
        return frames#(img_group, label)

    def __len__(self):
        return self.datalen

if __name__ == "__main__":
    #Find the corresponding frame_id and uid from vido_id and frame.
    id_dict = pd.read_csv('/export/data/dataset/EPIC-KITCHENS/annotations-master/EPIC_train_action_labels.csv')
    video_id = 'P01_01'
    frame = 15811
    idx = id_dict[(id_dict["video_id"]==video_id) & (id_dict["stop_frame"]>=frame) & (id_dict["start_frame"]<=frame)]
    frame_id = frame - idx['start_frame']
    frame_id = frame - idx.at[idx.index[0],'start_frame']
    uid = idx.at[idx.index[0], 'uid']
    dataloader = EpicDataset()
    dataloader[uid, frame_id]
