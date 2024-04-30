import numpy as np
import os
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList, Sequence_RGBT
from pytracking.utils.load_text import load_text

class VTUAVSTDataset(BaseDataset):
    # LasHeR dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vtuavst_path
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
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'vtuavst', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)
        
    def _get_sequence_list(self): # 
        sequence_list= ['car_027', 'tricycle_006', 'car_022', 'elebike_004', 'car_056', 'pedestrian_028', 'pedestrian_148', 'truck_008', 'pedestrian_080', 'car_006', 'pedestrian_162', 'car_012', 'car_129', 'pedestrian_056', 'pedestrian_196', 'pedestrian_088', 'tricycle_011', 'pedestrian_079', 'pedestrian_025', 'pedestrian_006', 'car_123', 'train_004', 'bike_005', 'pedestrian_026', 'tricycle_010', 'pedestrian_050', 'pedestrian_019', 'pedestrian_058', 'car_079', 'pedestrian_173', 'bus_010', 'pedestrian_127', 'pedestrian_234', 'pedestrian_036', 'car_077', 'truck_004', 'c-vehicle_003', 'car_106', 'car_005', 'car_112', 'pedestrian_211', 'car_067', 'pedestrian_060', 'car_101', 'car_059', 'tricycle_019', 'elebike_032', 'pedestrian_044', 'pedestrian_109', 'pedestrian_150', 'bus_026', 'pedestrian_034', 'pedestrian_142', 'tricycle_003', 'pedestrian_227', 'car_007', 'pedestrian_130', 'pedestrian_051', 'pedestrian_143', 'pedestrian_230', 'tricycle_016', 'elebike_031', 'pedestrian_017', 'car_042', 'pedestrian_093', 'pedestrian_023', 'pedestrian_113', 'tricycle_032', 'tricycle_027', 'pedestrian_192', 'pedestrian_149', 'bike_008', 'car_096', 'elebike_018', 'pedestrian_213', 'tricycle_023', 'pedestrian_095', 'car_109', 'pedestrian_229', 'tricycle_007', 'pedestrian_179', 'pedestrian_139', 'pedestrian_007', 'bus_029', 'pedestrian_215', 'pedestrian_053', 'car_020', 'elebike_007', 'pedestrian_112', 'pedestrian_041', 'car_064', 'car_053', 'pedestrian_209', 'car_060', 'car_063', 'pedestrian_119', 'pedestrian_001', 'pedestrian_185', 'pedestrian_027', 'car_061', 'pedestrian_134', 'pedestrian_156', 'pedestrian_122', 'pedestrian_117', 'tricycle_004', 'pedestrian_154', 'tricycle_005', 'car_072', 'pedestrian_183', 'tricycle_017', 'pedestrian_033', 'elebike_002', 'bus_019', 'pedestrian_064', 'excavator_001', 'pedestrian_015', 'pedestrian_195', 'car_097', 'pedestrian_161', 'pedestrian_010', 'ship_001', 'elebike_006', 'bus_028', 'bike_006', 'pedestrian_120', 'pedestrian_121', 'pedestrian_152', 'pedestrian_153', 'elebike_011', 'pedestrian_052', 'bus_012', 'car_110', 'car_065', 'pedestrian_077', 'pedestrian_111', 'bus_021', 'elebike_019', 'pedestrian_163', 'pedestrian_005', 'pedestrian_110', 'bus_006', 'bus_014', 'pedestrian_151', 'pedestrian_217', 'pedestrian_164', 'bus_001', 'pedestrian_089', 'bus_007', 'pedestrian_136', 'elebike_008', 'pedestrian_016', 'tricycle_008', 'pedestrian_098', 'pedestrian_062', 'animal_001', 'car_004', 'bike_003', 'tricycle_009', 'car_132', 'car_049', 'cable_002', 'pedestrian_046', 'elebike_005', 'truck_007', 'pedestrian_038', 'pedestrian_155', 'bus_004', 'pedestrian_138', 'car_128', 'elebike_010', 'tricycle_035', 'pedestrian_020', 'train_003', 'tricycle_037', 'car_075', 'pedestrian_123']
        # sequence_list = list(reversed([sequence_list[i:i+35] for i in range(0,len(sequence_list),35)][0]))[:3]     # split to test
        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = self.base_path+sequence_info["name"]
            #sequence_info["path"] = '/data/VTUAV/train/'+sequence_info["name"]            
            #sequence_info["startFrame"] = int('1')
            #print(end_frame[i])
            #sequence_info["endFrame"] = end_frame[i]
                
            #sequence_info["nz"] = int('6')
            #sequence_info["ext"] = 'jpg'
            sequence_info["anno_path"] = sequence_info["path"]+'/rgb.txt'
            #sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
    