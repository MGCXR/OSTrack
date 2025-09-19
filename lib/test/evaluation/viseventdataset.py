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
        