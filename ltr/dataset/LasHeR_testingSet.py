import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from ltr.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from ltr.admin.environment import env_settings

class LasHeR_testingSet(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        self.root = env_settings().LasHeR_testingSet_dir if root is None else root
        super().__init__('LasHeR_testingSet', root, image_loader)

        # video_name for each sequence
        self.sequence_list = ['10runone', '11leftboy', '11runtwo', '1blackteacher', '1boycoming', '1stcol4thboy', '1strowleftboyturning', '1strowrightdrillmaster', '1strowrightgirl3540', '2girl', '2girlup', '2runseven', '3bike1', '3men', '3pinkleft', '3rdfatboy', '3rdgrouplastboy', '3thmoto', '4men', '4thboywithwhite', '7rightorangegirl', 'AQgirlwalkinrain', 'AQtruck2north', 'ab_bikeoccluded', 'ab_blkskirtgirl', 'ab_bolstershaking', 'ab_girlchoosesbike', 'ab_girlcrossroad', 'ab_pingpongball2', 'ab_rightlowerredcup_quezhen', 'ab_whiteboywithbluebag', 'advancedredcup', 'baggirl', 'ballshootatthebasket3times', 'basketball849', 'basketballathand', 'basketboy', 'bawgirl', 'belowdarkgirl', 'besom3', 'bike', 'bike2left', 'bike2trees', 'bikeboy', 'bikeboyintodark', 'bikeboyright', 'bikeboyturn', 'bikeboyturntimes', 'bikeboywithumbrella', 'bikefromlight', 'bikegoindark', 'bikeinrain', 'biketurnright', 'blackboy', 'blackboyoncall', 'blackcarturn', 'blackdown', 'blackgirl', 'blkboy`shead', 'blkboyback', 'blkboybetweenredandwhite', 'blkboydown', 'blkboyhead', 'blkboylefttheNo_21', 'blkboystand', 'blkboytakesumbrella', 'blkcaratfrontbluebus', 'blkgirlumbrella', 'blkhairgirltakingblkbag', 'blkmoto2north', 'blkstandboy', 'blktribikecome', 'blueboy', 'blueboy421', 'bluebuscoming', 'bluegirlbiketurn', 'bottlebetweenboy`sfeet', 'boy2basketballground', 'boy2buildings', 'boy2trees', 'boy2treesfindbike', 'boy`headwithouthat', 'boy`sheadingreycol', 'boyaftertree', 'boyaroundtrees', 'boyatdoorturnright', 'boydownplatform', 'boyfromdark', 'boyinlight', 'boyinplatform', 'boyinsnowfield3', 'boyleftblkrunning2crowd', 'boylefttheNo_9boy', 'boyoncall', 'boyplayphone', 'boyride2path', 'boyruninsnow', 'boyscomeleft', 'boyshead9684', 'boyss', 'boytakingbasketballfollowing', 'boytakingplate2left', 'boyunder2baskets', 'boywaitgirl', 'boywalkinginsnow2', 'broom', 'carbehindtrees', 'carcomeonlight', 'carcomingfromlight', 'carcominginlight', 'carlight2', 'carlightcome2', 'caronlight', 'carturn117', 'carwillturn', 'catbrown2', 'catbrownback2bush', 'couple', 'darkcarturn', 'darkgirl', 'darkouterwhiteboy', 'darktreesboy', 'drillmaster1117', 'drillmasterfollowingatright', 'farfatboy', 'firstexercisebook', 'foamatgirl`srighthand', 'foldedfolderatlefthand', 'girl2left3man1', 'girl`sblkbag', 'girlafterglassdoor', 'girldownstairfromlight', 'girlfromlight_quezhen', 'girlinrain', 'girllongskirt', 'girlof2leaders', 'girlrightthewautress', 'girlunderthestreetlamp', 'guardunderthecolumn', 'hugboy', 'hyalinepaperfrontface', 'large', 'lastleftgirl', 'leftblkTboy', 'leftbottle2hang', 'leftboy2jointhe4', 'leftboyoutofthetroop', 'leftchair', 'lefterbike', 'leftexcersicebookyellow', 'leftfarboycomingpicktheball', "leftgirl'swhitebag", 'lefthyalinepaper2rgb', 'lefthyalinepaperfrontpants', 'leftmirror', 'leftmirrorlikesky', 'leftmirrorside', 'leftopenexersicebook', 'leftpingpongball', 'leftrushingboy', 'leftunderbasket', 'leftuphand', 'littelbabycryingforahug', 'lowerfoamboard', 'mandownstair', 'manfromtoilet', 'mangetsoff', 'manoncall', 'mansimiliar', 'mantostartcar', 'midblkgirl', 'midboyNo_9', 'middrillmaster', 'midgreyboyrunningcoming', 'midof3girls', 'midredboy', 'midrunboywithwhite', 'minibus', 'minibusgoes2left', 'moto', 'motocomeonlight', 'motogoesaloongS', 'mototaking2boys306', 'mototurneast', 'motowithbluetop', 'pingpingpad3', 'pinkwithblktopcup', 'raincarturn', 'rainycarcome_ab', 'redboygoright', 'redcarcominginlight', 'redetricycle', 'redmidboy', 'redroadlatboy', 'redtricycle', 'right2ndflagformath', 'right5thflag', 'rightbike', 'rightbike-gai', 'rightblkboy4386', 'rightblkboystand', 'rightblkfatboyleftwhite', 'rightbluewhite', 'rightbottlecomes', 'rightboy504', 'rightcameraman', 'rightcar-chongT', 'rightcomingstrongboy', 'rightdarksingleman', 'rightgirltakingcup', 'rightwaiter1_quezhen', 'runningcameragirl', 'shinybikeboy2left', 'shinycarcoming', 'shinycarcoming2', 'silvercarturn', 'small-gai', 'standblkboy', 'swan_0109', 'truckgonorth', 'turning1strowleft2ndboy', 'umbreboyoncall', 'umbrella', 'umbrellabyboy', 'umbrellawillbefold', 'umbrellawillopen', 'waitresscoming', 'whitebikebelow', 'whiteboyrightcoccergoal', 'whitecarcomeinrain', 'whitecarturn683', 'whitecarturnleft', 'whitecarturnright', 'whitefardown', 'whitefargirl', 'whitegirlinlight', 'whitegirltakingchopsticks', 'whiteofboys', 'whiteridingbike', 'whiterunningboy', 'whiteskirtgirlcomingfromgoal', 'whitesuvturn', 'womanback2car', 'yellowgirl118', 'yellowskirt']


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'LasHeR_testingSet'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}
    def get_num_sequences(self):
        return len(self.sequence_list)
    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'visible', sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'infrared', sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_list_v = [self._get_frame_v(seq_path, f) for f in frame_ids]
        frame_list_i = [self._get_frame_i(seq_path, f) for f in frame_ids]
        frame_list  = frame_list_v + frame_list_i # 6
        #print('get_frames frame_list', len(frame_list))
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta
