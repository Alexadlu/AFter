import numpy as np
from lib.test.evaluation.data import Sequence, Sequence_RGBT, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class DepthTracktestingSetDataset(BaseDataset):
    # DepthTrack dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.depthtracktestingset_path
        print(self.base_path)
        self.sequence_list = self._get_sequence_list()
        '''
        self.sequence_list = [
            {
                'name': 序列名1,
                'path': '/home/liulei/Datasets/LasHeR/序列名1',
                'anno_path': '/home/liulei/Datasets/LasHeR/序列名1/init.txt'
            },
            {
                'name': 序列名2,
                'path': '/home/liulei/Datasets/LasHeR/序列名2',
                'anno_path': '/home/liulei/Datasets/LasHeR/序列名2/init.txt'
            },
            ...
        ]
        '''

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        '''
        sequence_info = {
                'name': 序列名1,
                'path': '/home/liulei/Datasets/LasHeR/序列名1',
                'anno_path': '/home/liulei/Datasets/LasHeR/序列名1/init.txt'
            }
        '''
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=[' ', '\t', ','], dtype=np.float64)
        
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'color')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_v = [os.path.join(sequence_path, 'color', img) for img in img_list_v]
        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'depth')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_i = [os.path.join(sequence_path, 'depth', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4: # False
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'depthtracktestingset', ground_truth_rect, m=['v', 'd'])

    def __len__(self):
        return len(self.sequence_list)
        
    def _get_sequence_list(self):
        sequence_list = os.listdir(self.base_path)

        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = os.path.join(self.base_path, sequence_info["name"])
            sequence_info["anno_path"] = os.path.join(sequence_info["path"], 'groundtruth.txt')
            sequence_info_list.append(sequence_info)
        return sequence_info_list
    
