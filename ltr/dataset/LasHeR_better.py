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

class LasHeR_better(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        self.root = env_settings().LasHeR_dir if root is None else root
        super().__init__('LasHeR_better', root, image_loader)

        # video_name for each sequence
        self.sequence_list = ['blkcarstart', 'whitebikebehind2', 'whiteboy', 'boyrunning', 'boywaitgirl', 'rightestblkboy2', 
        'midflag-qzc', 'whiteatright', 'caronlight', 'left4thblkboyback', 'abeauty_1202', 'carlightcome2', 'redcupatleft', 
        'boywalkinginsnow', 'the4thwhiteboy', 'sisterswithbags', 'mototurn', 'Abluemotoinrain', 'belowyellow-gai', 'biketonorth', 
        'rightgreen', 'runningcameragirl', 'darkleftboy2left', 'whitecarcoming', 'blkridesbike', 'yellowgirl118', 'redetricycle', 
        'exercisebook', 'redgirl1497', 'bikeinhand', 'rightexercisebookwillfly', 'bikeboystrong', 'whiteboyright', 'whiteboyphone',
        'aboyleft_1202', 'etrike', 'boywalkinginsnow3', 'minibus', '10crosswhite', 'boytakepath', 'rightbottle', 'goaftrtrees', 
        '3thmoto', 'carturnleft', 'whitegirl_0115', 'leftrushingboy', 'blkboytakesumbrella', 'carstop', 'bus2north', 
        'truckgonorth', 'girlintrees', 'greenboyafterwhite', '10rightboy', 'boysumbrella3', 'blkboyback636', 'redshirtman', 
        'leftof2girls', 'leftredcup', 'ab_mototurn', 'righttallholdball', '10rightblackboy', 'bikeblkturn', '7rightredboy', 
        'blackbag', 'whitecarcomes', 'girlwithumbrella', 'camonflageatbike', 'bikegoindark', 'bikeboygo', 'manupstairs', 
        'whitecatjump', 'carstart189', 'biketurndark', 'whitebikebehind', 'lefthandfoamboard', 'motowithtopcoming', 
        'girlwithredhat', 'redtricycle', 'bike2trees86', 'whiteboycome', 'yellowgirlwithbowl', 'left3rdgirlbesideswhitepants', 
        'blueumbrellagirl', 'girlthroughtrees', 'moto', 'leftblkTboy', 'midboyNo_9', 'mototakinggirl', '2ndcarcome', 'leftmirror',
        'man_0109', 'midwhitegirl', 'boyinplatform', 'guardman', 'manwithyellowumbrella', 'twoperson_1202', 'lightmotocoming', 
        'blackbagbike', 'blkstandboy', 'boyoncall', 'motogoesaloongS', 'ajiandan_catwhite', 'boy2trees', 'carturn',
        'Amotowithbluetop', 'whitecar2west', 'girlalone', 'whitecarstart183', 'boyshorts', 'bikeboy', 'bikecoming176', 
        'doginrain', 'rainblackcarcome', 'lowerfoamboard', 'man', 'browncar2east', 'blackbaggirl', 'motocometurn', 
        'rightcar-chongT', 'rightgreenboy', 'takeoutman953', 'The_one_on_the_left_in_black_1202', 'frontmirror', 
        'rightholdball1096', '3whitemen', 'outer2leftmirrorback', 'e-tricycle', 'manatmoto', 'bike2trees', 'browncarturn',
        '7rightorangegirl', 'carleaves', 'blackboy186', 'whitecarturn2', 'boydownplatform', 'rightwhitegirl',
        'whitegirlcoming', '1blackteacher', 'bikeboyturn', 'rightblkboybesidesred', 'bikeboy173', 'ab_bolster',
        'skirtwoman', 'boy_0109', 'man_with_black_clothes3', 'motolightturnright', 'bike2', 'boybikeblueumbrella',
        'large', 'catbrownback2bush', 'rightboywitjbag', 'ab_moto2north0', 'lover_1202', 'shunfengtribike', 'umbrellabikegirl',
        'leftorangeboy', 'carleaveturnleft', 'redwhitegirl', 'biketurnright', 'womanback2car', 'blkboy198', 'blackgirl', 
        'redgirl2trees', 'suitcase', 'carturn117', 'bookatfloor', 'Aboydownbike', 'motowithbluetop', 'blkboycoming',
        'rightgirlbikecome', 'blkhairgirltakingblkbag', 'boyrightthelightbrown', 'Aredtopmoto', 'manfromcar', 'whitecarback', 
        'whitecarturn137', 'yellowcar', 'truckk', 'girl', 'Agirlrideback', 'elegirl', 'whitegirlundertheumbrella', 'carout', 
        'carstarts', 'blackcarturn', 'ab_hyalinepaperatground', 'takeoutmototurn', 'man2startmoto', 'openningumbrella', 
        'leftexcersicebookyellow', 'farwhitecarturn', 'whitecarinrain', 'blueboy', 'carbesidesmoto', '2ndgirlmove', 'bluemanof3', 
        'The_girl_with_the_cup_1202', 'bluegirlbiketurn', 'leftclosedexersicebook', 'Amidredgirl', 'manbikecoming', 'whitegirl2', 
        'manfromtoilet', 'takeoutmanleave', '2girlsridebikes', 'blkboybike', 'girltakemoto', 'ab_blkskirtgirl', 'nearmangotoD', 
        'carcominginlight', 'easy_blackboy', 'blackboy', 'girlunderthestreetlamp', 'folddenumbrellainhand', 'leftwhitebike', 
        'treeboy', 'motowithtop', 'whiteboy242', 'right2ndflagformath', 'mototurn134', 'bluelittletruck', 'umbrellawillbefold', 
        'cycleman', 'motowithgood', 'AQraincarturn2', 'umbrellabyboy', 'boyshead9684', 'waitresscoming', 'carturnleft109', 
        'runningwhiteboy249', 'lightcarcome', 'girltakingplate', 'openthisexersicebook', 'moto78', 'bikecoming', 'takeoutmoto', 
        'blkboydown', 'midgirl', 'manoncall', 'boywithumbrella', 'mototurntous', 'bigbus', 'boybetween2blkcar', 'manrun250',
        'rightboywithwhite', 'threepeople', 'mototake2boys123', 'fardarkboyleftthe1stgirl', 'raincarstop', 'basketboyblack',
        'blkcarinrain107', 'shinycarcoming2', 'the4thboystandby', 'boyridesbike', 'occludedmoto', 'take-out-motocoming',
        'boyaftertree', 'schoolofeconomics-yxb', 'boytakebox', 'leftblkboy648', '11righttwoboy', '4thboywithwhite', 
        'silvercarcome', '7one', 'downwhite_1227', 'bus', 'car2north', 'whitebetweenblackandblue', 'smallmoto', 
        'manglass1', 'whitecarturnl', 'boystandinglefttree', 'tallboyNumber_9', 'basketboywhite', 'blkboy`shead', 
        '4men', 'blackcargo', 'rightblkgirlNo_11', 'lowerfoam2throw', 'leftgirl1299', 'guardatbike_ab', 'girltakebag', 
        'dogforward', 'whiteridingbike', 'whiteboybike', 'boyalone', '1righttwogreen', 'AQrightofcomingmotos', 'whitedown', 
        'e-tribike', 'girlsheadwithhat', 'blackdown', 'mirroratleft', 'whiteboywait', 'moto2north1', 'left2flagfornews', 
        'leftlightsweaterboy', 'besom3', 'whitemancome', 'whitecargo', 'leftdress-gai', 'boytakingbasketballfollowing',
        'stronggirl', 'bikewithbag', 'boybesidesblkcarrunning', 'mantoground', 'boyscomeleft', 'midblkgirl', 'leftbasketball', 
        'huggirl', 'blkboywithbluebag', 'bikeboywithumbrella', 'manwalkincars', 'Ahercarstart', 'boyaroundtrees', 'carfromnorth2', 
        'righttallnine-gai', 'umbreboyoncall', 'girlwithblkbag', 'manaftercars', 'minibusback', 'boyouttrees', 'boyrun', 
        'AQmanfromdarktrees', 'runningwhiteboy', 'boywithshorts2', '5numone', 'pingpongpad2', 'boyride2path', 'umbrellainred',
        'minibuscome', 'motocome122', 'guardunderthecolumn', 'whitecarturnl2', 'whitegirlcrossingroad', 'actor_1202', 
        'yellowskirt', 'blklittlebag', 'carcomeonlight2', 'ab_boyfromtrees', 'minibus152', 'wanderingly_1202', '4boys2left',
        'rightredboy', 'ab_leftmirrordancing', 'AQwhitetruck', 'besom2-gai', 'drillmaster', 'girl2-gai', 'boyrightrubbish', 
        'blackinpeople', 'blkteacher`shead', 'ab_motoinrain', 'drillmaster1117', 'blkgirlumbrella', 'boyleft', 
        'rightboywithbluebackpack', 'bluegirlriding', 'blkcarcome115', 'boyinsnowfield2', 'boyplayphone', 'gonemoto_ab',
        'farredcar', 'boy2_0115', 'greenboywithgirl', 'boysitbike', '2girl', 'jeepblack', 'rightbike', 'girlumbrella', 
        '2boysatblkcarend', 'car2north3', 'leftgirlunderthelamp', 'girlblack', 'whitecarturn85', 'blkcarinrain', 'hatboy`shead', 
        'redboywithblkumbrella', 'whitecarstart126', 'girlpickboy', 'motocome', 'motocoming', 'blkgirlbike', '3rdfatboy', 
        'turnblkbike', 'nikeatbike', 'boyinsnowfield4', 'motowithblack', 'rainysuitcase', 'folderatlefthand', 'midpinkblkglasscup',
        'firstexercisebook', '2boys', 'manfromcar302', 'whitesuvstop', 'carlight2', 'blkboyatbike', 'bikeorange', 
        'blkboywithwhitebackpack', 'bikeinrain', 'rainywhitecar', 'bike150', 'bus2north111', 'motoinrain56', 
        'greengirls', 'whitebikebehind172', 'blackman2', 'Athe3rdboybesidescar', 'mangetsoff', 'whiteskirtgirl',
        'ab_righthandfoamboard', 'girlruns2right', 'girlinrain', 'Awhitecargo', 'camera_1202', 'drillmasterfollowingatright', 
        'blackridebike', 'manphone', 'rightgirlwithumbrella', 'mototurneast', 'pinkgirl285', 'whitegirl', 'carwillturn',
        'mirrorfront', 'right2ndfarboytakinglight2left', '10phone_boy', 'boydown', 'browncar2north', 'Awhiteleftbike', 
        'boysumbrella', 'redminirtruck', 'pinkwithblktopcup', 'mototurnleft', 'leftmirrorlikesky', 'rightof2boys',
        'motocomeinrain', '2up', 'blackpantsman', 'Take_an_umbrella_1202', 'darkgirl', 'manbesideslight', 'whiterunningboy', 
        'blkcarcome155', 'mantostartcar', 'greenboy438', 'carstart', 'boygointrees', 'leftthrowfoam', 'lightcarfromnorth',
        'comebike', 'greenleftbackblack', 'pingpongpad', 'womanstartbike', 'ab_leftfoam', 'redcarcominginlight', 'bikefromlight',
        'singleboywalking', '2ndbikecoming', 'tallwhiteboy', 'whiteaftertrees', 'whitecarturn178', '1boycoming', 'girlaftertree', 
        'girlgoleft', 'boyplayingphone366', '10runone', 'battlerightblack', 'rainycarcome_ab', 'boy_0115', 'girlinfrontofcars', 
        '3men', 'blkmoto', 'whiteboyatbike', 'large2-gai', 'redmotocome', 'manafetrtrees', 'motoman', 'whitecarturn', 
        'rightgirlatbike', 'bikeboycome', 'downmoto', '2boyscome', 'collegeofmaterial-gai', 'girldownstairfromlight', 
        'rightboy`shead', 'minibus125', 'blueboy421', 'Awoman2openthecardoor', 'motofromdark', 'boybackpack', 'leftcup', 
        'woman', 'left2ndboy', 'girlrightthewautress', 'takeoutmoto521', '2outdark', 'umbrellainyellowhand', 'man_with_black_clothes2', 'advancedredcup', 'blackboyoncall', 'motolight', '3girl1', 'Acarlightcome', 'leftchair', 'leftgirlchecking', 'redbackpackgirl', 'raincarstop2', 'rightblkboy', 'righthunchblack', 'boyshead', 'blackcar126', 'whitecarturnright', 'basketballshooting', 'bike', 'bikeout', '11runone', 'cameraman_1202', 'AQboywithumbrella415', 'whitecarleave', '2girlgoleft', 'boyscome', 'blktribikecome', 'comecar', 'blkboylefttheNo_21', 'womanopendoor', 'whiteboy395', '6walkgirl', 'girlbesidesboy', 'blkmotocome', 'midboyplayingphone', 'womanongrass', 'AQwhiteminibus', 'shinycarcoming', 'manglass2', 'belowdarkgirl', 'whitegirl209', 'rightboy479', 'boymototakesgirl', 'girlshakeinrain', 'girlturnbike', 'rightgirl', 'carcomeonlight', 'umbellaatnight', 'blkboyonleft', 'leftboy', 'Ablkboybike77', '1boygo', 'ab_girlcrossroad', 'blkbikecomes', 'lightcarstop', 'whitecarcome192', 'ballshootatthebasket3times', 'bluebuscoming', 'farwhiteboy', 'notebook-gai', 'moto2north', 'leftgirlafterlamppost', 'rightblkgirlrunning', 'girloutreading', 'leftredflag-lsz', 'girlbike156', 'bikeboy128', 'whiteofboys', 'blktakeoutmoto', 'boysumbrella2', 'rightcameraman', 'whitecarturn683', 'blackcarcome', 'boyinsnowfield3', 'whiteskirtgirlcomingfromgoal', 'foamboardatlefthand', 'manstarttreecar', 'boywithblkbackpack', 'blackphoneboy', 'bikeumbrellacome', 'twopeopleelec-gai', 'leftaloneboy-gai', 'leftdrillmasterstandsundertree', 'girlfoldumbrella', 'blackcarturn183', 'boywalkinginsnow2', 'boybehindbus', 'whitecarcome', 'blkcarcomeinrain', 'nightmototurn', 'midboy', 'boyunderthecolumn', 'midof3girls', 'easy_whiterignt2left', 'blackof4bikes', 'whitecat', 'boyride2trees', 'blackman_0115', 'man_head', 'whitegirl2_0115', 'blkcarfollowingwhite', 'boy2buildings', 'trimototurn', 'girlplayingphone', 'ab_redboyatbike', 'blackboypushbike', 'whitegirlatbike', 'bluetruck', 'blkboylefttheredbagboy', 'rightblkboy2', 'AQgirlwalkinrain', 'darkgiratbike', 'blackcar131', 'orangegirl', 'girlridesbike', 'ab_bikeboycoming', 'girlleaveboys', 'leftmirrorshining', 'blacktallman', 'whitebikebelow', 'dogfollowinggirl', 'darktreesboy', 'boyputtinghandup', 'rightblkboystand', 'rightmirrorlikesky', 'large3-gai', '3rdboy', 'umbregirl', 'boyruninsnow', 'rightwhite_1227', 'boyunder2baskets', 'ab_bolstershaking', 'bluemanatbike', 'midredboy', 'boyleave', 'midblkboyplayingphone', 'whitecarafterbike', 'boybesidescarwithouthat', 'whiteTboy', 'yellowatright', 'ab_catescapes', 'girlbike', 'whitacatfrombush', 'lonelyman', 'pinkbikeboy', 'AQtaxi', 'AQmidof3boys', 'boybikewithbag', 'agirl', 'manopendoor', 'couple', 'rightboyatwindow', 'rightwhitegirlleftpink', 'boyblackback', 'pickuptheyellowbook', 'boyright', 'mansimiliar', 'motoinrain', 'redgirlsits', 'ab_girlchoosesbike', 'meituanbike', '2boyscome245', 'blkcargo', 'silvercarturn', 'bawgirl', 'nearestleftblack', 'whiteboywithbag', 'girlleft2right2', 'rightestblkboy', 'blackcarturn175', 'easy_4women', 'leftblkboy', 'Aab_whitecarturn', 'boyfromdark', 'redcar', 'manwait1', 'umbrellawillopen', 'bikefromnorth2', 'small2-gai', 'yellowexcesicebook', 'motobesidescar', 'schoolbus', 'small-gai', 'boyturn', 'girlleft2right1', 'girllongskirt', 'truckwhite', 'AQmotomove', 'left11', 'blkcar2north', 'girlatwindow', 'boyshead2', 'The_girl_back_at_the_lab_1202', 'moto2', 'motosmall', 'whitewoman', 'AQgirlbiketurns', 'motocomeonlight', 'ab_bikeoccluded', 'bikeboyleft', '2sisiters', 'carfromnorth', 'girltakingmoto', 'blkboywithblkbag', 'leftblackboy', 'boylefttheNo_9boy', 'bord', 'mototaking2boys', '2ndbus', '2girlinrain', 'carclosedoor', '1whiteteacher', 'boyinlight', 'carfarstart', '2boysup', 'right5thflag', 'boycomingwithumbrella', 'truck', 'foamatgirl`srighthand', 'checkedshirt', 'ab_minibusstops', 'thefirstexcersicebook', 'rightbluegirl', 'darkredcarturn', 'whiteof2boys', "leftgirl'swhitebag", 'boyss', 'manaftercar114', 'whitefardown', 'rightmirrornotshining', 'ab_girlrideintrees', 'whiteboyback', 'blackboy256', 'moto2north2', 'whitecarstart', 'boycome', 'blueboybike', '3bike2', 'manatwhiteright', '2gointrees', 'pinkgirl', 'whiteboyup', 'pingpingpad3', 'leftmen-chong1', 'AQblkgirlbike', 'nearstrongboy', 'bluegirl', 'blkbikefromnorth', 'rightblkgirl', 'rightboywithbackpackandumbrella', 'ab_shorthairgirlbike', 'leftopenexersicebook', 'basketboy', 'biketurnleft', 'easy_runninggirls', 'rightredboy954', 'leftuphand', 'boyumbrella4', 'darkcarturn', 'dogouttrees', 'blkmoto2north', 'bluecar', 'blackcar', 'blackcarback', 'firstrightflagcoming', 'whitesuvcome', 'boyrideoutandin', 'takeoutman', 'redumbrellagirlcome', 'blackcarcoming', 'umbrellainblack', 'cargirl2', 'whitemotoout', 'boysback', 'Amotoinrain150', 'whiteman', 'blkboywithglasses', 'redbaginbike', 'dotat43', 'stubesideswhitecar', 'bikecome', 'motocross', 'leftgirl', 'girlshead', '2whitegirl', 'whitewoman_1202', 'whitegirlinlight', 'bike2north', 'whitegirlwithumbrella', 'moto2north101', '2boysbesidesblkcar', '7rightwhitegirl', 'whiteaftertree', 'leftunderbasket', 'manrun', 'boyshead509', 'rightboystand', 'whitecarturnright248', 'redup', 'elector_1227', 'carstart2east', 'motostraught2east', 'outerfoam', 'meituanbike2', 'yellowtruck', 'midgreyboyrunningcoming', 'theleftboytakingball', 'firstboythroughtrees', 'moto2trees2', 'stripeman', 'whiteboy`head', '3bike1', 'unbrellainbike', 'motoprecede', 'rightboy_1227', 'whitecarcomeinrain', 'righthand`sfoam', 'whiteshirt', 'mototake2boys', 'the2ndboyunderbasket', 'blackturnr', 'bluemoto', '2rdcarcome', 'whitecarleave198', 'boyleft161', 'whitemoto', 'whiteminibus197', 'redgirl', 'boy2treesfindbike', 'motocome2left', 'lightredboy', 'AQbikeback', 'baggirl', 'biketurnleft2', 'blkman2trees', 'boy1227', 'whitegirltakingchopsticks', 'rightwaiter1_quezhen', 'tallboyblack', 'mototurnright', 'minibusgoes2left', 'blkboy', 'catbrown', '4sisters', 'whitecar', 'car', '2rdtribike', 'blkumbrella', 'basketball', 'basketman', 'bluebike', '3blackboys', 'biked', 'standblkboy', 'bikeboyright', 'motoslow', 'carturncome', 'trashtruck', 'rightredboy1227', 'blueboycome', 'boyfromdark2', 'ajiandan_blkdog', 'greenfaceback', 'belowrightwhiteboy', 'whitefargirl', 'whitecarfromnorth', 'blkcarcome', 'rightstripeblack', 'girlbikeinlight', 'midblkbike', 'whitecar70', 'greenboy', 'catbrown2', 'blkskirtwoman', 'bikefromnorth257', 'blkmaninrain', 'blkboywillstand', 'motobike', 'motocomenight', 'boyridesbesidesgirl', 'ab_rightmirror', 'rightshiningmirror', 'girl2left3man1', 'whitecarturnleft', 'leftmirror2', 'whiteboycome598', 'redcarturn', 'twopeople', 'motocominginlight', 'blkboywithumbrella', 'mancarstart', 'mototurn102', 'blueboyopenbike', 'blackridebike2', 'girl2trees', 'boywithshorts', 'womanaroundcar', 'bikeblkbag', 'elector_0115', 'yellowgirl', 'mancrossroad', 'midboyblue', 'girlcoat', 'girlatleft', 'manaftercar', 'boytakingcamera', 'rightgirlbike', 'biketurn', 'yellowumbrellagirl', 'ab_whiteboywithbluebag', 'whiteblcakwoman', 'redroadlatboy', 'whitesuvturn', 'lefthyalinepaper', 'bikefromwest', 'moto2west', 'rightmirrorbackwards', 'ab_motocometurn', '9whitegirl', 'AQtruck2north', 'whitegirl1227', 'bikefromnorth', 'boytakesuicase', 'rightholdball', 'bus2', 'ninboy-gai', 'bike2left', 'blueboy85', 'umbrella', "comingboy'shead", 'boyplayingphone', 'moto2trees', 'raincarturn', 'bikeboyturntimes', 'boytakingplate2left', 'right4thboy', 'blkmototurn', 'oldwoman', 'leftmirrorside', 'rightblkboy4386', 'farfatboy', 'blackmanleft', 'hugboy', 'rightblkboy188', 'jeepleave', 'truckcoming', 'blkcaratfrontbluebus', 'lightcarstart', 'carlight', 'leftshortgirl', 'blueboywalking', 'fatmancome', 'car2north2']

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'LasHeR_better'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_name):
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'visible', sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'infrared', sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)

    def get_frames(self, seq_name, frame_ids, anno=None):
        seq_path = os.path.join(self.root, seq_name)
        frame_list_v = [self._get_frame_v(seq_path, f) for f in frame_ids]
        frame_list_i = [self._get_frame_i(seq_path, f) for f in frame_ids]
        frame_list  = frame_list_v + frame_list_i # 6
        if seq_name not in self.sequence_list:
            print('warning!!!'*100)
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
