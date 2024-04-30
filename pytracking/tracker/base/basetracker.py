from _collections import OrderedDict
from ltr.models.transformer.position_encoding import PositionEmbeddingSine
class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params
        self.visdom = None

        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=256//2, sine_type='lin_sine',
                                                avoid_aliazing=True, max_spatial_resolution=18)  
    def predicts_segmentation_mask(self):
        return False


    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError


    def track(self, image, info: dict = None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')