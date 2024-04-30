import numpy as np
import os
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList, Sequence_RGBT
from pytracking.utils.load_text import load_text

class VTUAVLTDataset(BaseDataset):
    # LasHeR dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vtuavlt_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=[' ', '\t','', ','], dtype=np.float64)
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'rgb')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_v = [os.path.join(sequence_path, 'rgb', img) for img in img_list_v]
        
        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'ir')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_i = [os.path.join(sequence_path, 'ir', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'vtuavlt', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)
        
    def _get_sequence_list(self): # 
        sequence_list= ['bike_001', 'pedestrian_014', 'animal_004', 'car_036', 'pedestrian_013', 'car_008', 'car_057', 'pedestrian_226', 'tricycle_036', 'pedestrian_208', 'pedestrian_137', 'car_018', 'pedestrian_055', 'car_095', 'pedestrian_145', 'car_127', 'pedestrian_218', 'tricycle_012', 'elebike_003', 'elebike_013', 'bus_032', 'pedestrian_037', 'car_054', 'tricycle_025', 'pedestrian_024', 'pedestrian_204', 'car_046', 'pedestrian_168', 'car_073', 'car_055', 'pedestrian_132', 'tricycle_002', 'tricycle_018', 'pedestrian_232', 'pedestrian_002', 'pedestrian_184', 'pedestrian_220', 'elebike_012', 'pedestrian_199', 'pedestrian_141', 'pedestrian_021', 'pedestrian_144', 'pedestrian_178', 'elebike_014', 'elebike_009', 'pedestrian_140', 'pedestrian_212', 'car_119', 'truck_001', 'elebike_027', 'car_125', 'pedestrian_205', 'car_015', 'pedestrian_182', 'pedestrian_188', 'pedestrian_201', 'pedestrian_190', 'car_070', 'car_091', 'pedestrian_207', 'elebike_029', 'pedestrian_009', 'tricycle_015', 'bus_025', 'car_103', 'car_010', 'pedestrian_219', 'tricycle_013', 'car_001', 'pedestrian_187', 'animal_003', 'pedestrian_214', 'pedestrian_194', 'pedestrian_221']

        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = self.base_path+sequence_info["name"]
            #sequence_info["startFrame"] = int('1')
            #print(end_frame[i])
            #sequence_info["endFrame"] = end_frame[i]
                
            #sequence_info["nz"] = int('6')
            #sequence_info["ext"] = 'jpg'
            sequence_info["anno_path"] = sequence_info["path"]+'/rgb.txt'
            #sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
    