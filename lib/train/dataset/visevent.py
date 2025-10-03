import os
import os.path
import torch
import numpy as np
import pandas
import csv
import re
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings

class VisEvent(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        root= env_settings().visevent_dir if root is None else root
        super().__init__('VisEvent', root, image_loader)
        self.sequence_list = self._get_sequence_list()
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'visevent_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'visevent_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self.class_list = self._get_class_list()
        
    def _get_sequence_list(self):
        sequence_list = [f.name for f in os.scandir(self.root) if f.is_dir()]
        sequence_list.sort()
        return sequence_list
    
    def _get_class_list(self):
        class_list = [f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]
        class_list.sort()
        return class_list
    
    def get_name(self):
        return 'visevent'
    
    def get_event(self):
        return False
    
    def get_vis(self):
        return True
    
    def get_sequences_in_class(self, class_name):
        return class_name 
    
    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])
    
    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)
    
    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}
    
    def _get_frame_names(self, seq_path):
        frames = [f for f in os.listdir(os.path.join(seq_path,"vis_imgs")) if f.endswith('.bmp')]
        frames = sorted(frames, key=lambda x: int(re.search(r'\d+', x).group()))
        return frames
    
    def _get_frame(self, seq_path, frame_id):
        frame_names=self._get_frame_names(seq_path)
        vis=os.path.join(seq_path, "vis_imgs", frame_names[frame_id])
        event=os.path.join(seq_path, "event_imgs", frame_names[frame_id])
        vis_img=self.image_loader(vis)
        event_img=self.image_loader(event)
        return [vis_img,event_img]
        
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]    
        
        obj_class = self.class_list[seq_id]
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list, anno_frames, object_meta
    
    
