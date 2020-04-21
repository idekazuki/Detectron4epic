import numpy as np
import torch
import pandas as pd

from gulpio import GulpDirectory

class EpicDataset_detectron2(torch.utils.data.Dataset):
    """
    Epic-kitchen video dataset loader.
    Construct the epic-kitchen video dataset loder for training and validation.
    """
    def __init__(self, path='/home/yanai-lab/ide-k/ide-k/epic/data/processed/gulp/rgb_train/', frame_size=1, class_type='noun'):
        """
        Construct the epic-kitchen video dataset loader.
        Args:
            path (str): video path for epic dataset in gulpio format.
            class_type (str): Options includes 'noun', 'verb', 'noun+verb'.
        """
        self.path = path
        self.class_type = class_type
        self.dataset = GulpDirectory(path)
        self.gdict = list(self.dataset.merged_meta_dict.keys())
        self.datalen = len(self.gdict)

    def __getitem__(self, val):
        """
        Given the video_id, frame number , return frame
        Args:
            val (tuple): includes video_id (int) and fram number (int)
        Return:
            frame (np.array): image data [height, width, channel]
        """
        uid, frame_id = val
        frames, meta = self.dataset[str(uid)]
        frame = frames[frame_id]
        frame = frame[:, :, ::-1]
        return frame

    def __len__(self):
        return self.datalen

