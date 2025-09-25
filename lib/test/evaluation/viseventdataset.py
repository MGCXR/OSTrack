import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class VisEventDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.visevent_path
        self.sequence_list = self._list_sequences(self.base_path)
    
    def __len__(self):
        return len(self.sequence_list)
    
    def _list_sequences(self, root):
        return [entry.name for entry in os.scandir(root) if entry.is_dir()]
    
    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(set, seq_name) for set, seq_name in self.sequence_list])
    
    def _construct_sequence(self, set, sequence_name):
        anno_path = os.path.join(self.base_path, set, "anno.txt")
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')
        
        frames_path = os.path.join(self.base_path, set, "frames")
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        
        return Sequence(sequence_name, frames_list, 'visevent', ground_truth_rect.reshape(-1, 4))
    
    def __len__(self):
        return len(self.sequence_list)