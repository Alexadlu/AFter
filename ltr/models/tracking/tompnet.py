import math,sys,os
import torch.nn as nn
from collections import OrderedDict

import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor
from ltr.admin import loading
import ltr.models.transformer.transformer as trans
import ltr.models.transformer.hierarchical_attention as han
import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads
import pdb
import torch

class AFter(nn.Module):
    """Base on the ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))
  
    def get_positional_encoding(self, feat):
        nframes, nseq, _, h, w = feat.shape

        mask = torch.zeros((nframes * nseq, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)

        return pos.reshape(nframes, nseq, -1, h, w)
    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""
        #pdb.set_trace()
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        train_imgs_v = train_imgs[:2,...]
        test_imgs_v = test_imgs[:1,...]
        train_imgs_t = train_imgs[2:,...]
        test_imgs_t = test_imgs[1:,...] 

        # Extract backbone features
        train_feat_v = self.extract_backbone_features(train_imgs_v.reshape(-1, *train_imgs_v.shape[-3:]))
        test_feat_v = self.extract_backbone_features(test_imgs_v.reshape(-1, *test_imgs_v.shape[-3:]))
        train_feat_t = self.extract_backbone_features(train_imgs_t.reshape(-1, *train_imgs_t.shape[-3:]))
        test_feat_t = self.extract_backbone_features(test_imgs_t.reshape(-1, *test_imgs_t.shape[-3:]))
        
        # Classification features
        train_feat_head_v = self.get_backbone_head_feat(train_feat_v)
        test_feat_head_v = self.get_backbone_head_feat(test_feat_v)
        train_feat_head_t = self.get_backbone_head_feat(train_feat_t)
        test_feat_head_t = self.get_backbone_head_feat(test_feat_t)

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head_v,train_feat_head_t,test_feat_head_v,test_feat_head_t, train_bb
                                            , *args, **kwargs)

        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=False):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = AFter(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net
    
@model_constructor
def tompnet50_rgbt(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, out_feature_dim=256, frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2'], nhead=8, num_encoder_layers=6,
              num_decoder_layers=6,fia_layers=1, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=False):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception
    #pdb.set_trace()
    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)
    #pdb.set_trace()
    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    han_fusion = han.DynamicFusionModule()

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor,modals_fusion=han_fusion)
    # ToMP network
    net = AFter(feature_extractor=backbone_net, head=head, head_layer=head_layer)

    usepretrain = True
    # usepretrain = False
    if usepretrain:
        models_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        pretrained_path = os.path.join(models_path, 'pretrained/tomp50.pth.tar')
        pretrainmodel = torch.load(pretrained_path)['net']
        net.load_state_dict(pretrainmodel, strict=False)
    return net

@model_constructor
def tompnet101(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
               final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet101(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)


    # load pretrained model

    # ToMP network
    net = AFter(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net
