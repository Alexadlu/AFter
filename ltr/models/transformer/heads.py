import torch
import torch.nn as nn
import ltr.models.layers.filter as filter_layer
from ltr.models.transformer.position_encoding import PositionEmbeddingSine
import pdb
def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers


class Head(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,modals_fusion,
                 separate_filters_for_cls_and_bbreg=False):
        super().__init__()
        self.transformer_fusion = modals_fusion
        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg
        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=256//2, sine_type='lin_sine',
                                                    avoid_aliazing=True, max_spatial_resolution=18)
    def get_positional_encoding(self, feat):
        nframes, nseq, _, h, w = feat.shape

        mask = torch.zeros((nframes * nseq, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)
        return pos.reshape(nframes, nseq, -1, h, w)        
    def forward(self, train_feat_v,train_feat_t, test_feat_v,test_feat_t, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3
        #pdb.set_trace()
        num_sequences = train_bb.shape[1]

        if train_feat_v.dim() == 5:
            train_feat_v = train_feat_v.reshape(-1, *train_feat_v.shape[-3:])
            train_feat_t = train_feat_t.reshape(-1, *train_feat_t.shape[-3:])            
        if test_feat_t.dim() == 5:
            test_feat_v = test_feat_v.reshape(-1, *test_feat_v.shape[-3:])        
            test_feat_t = test_feat_t.reshape(-1, *test_feat_t.shape[-3:])

        # Extract features
        train_feat_v = self.extract_head_feat(train_feat_v, num_sequences)
        test_feat_v = self.extract_head_feat(test_feat_v, num_sequences)
        train_feat_t = self.extract_head_feat(train_feat_t, num_sequences)
        test_feat_t = self.extract_head_feat(test_feat_t, num_sequences)
        [f,b,c,w,h]=test_feat_v.shape
        
        train_feat1 = self.transformer_fusion(train_feat_v[0],train_feat_t[0]).unsqueeze(0)
        train_feat2 = self.transformer_fusion(train_feat_v[1],train_feat_t[1]).unsqueeze(0)
        train_feat = torch.cat([train_feat1,train_feat2],0)
        test_feat = self.transformer_fusion(test_feat_v[0],test_feat_t[0]).unsqueeze(0)

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)


class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0) # (nf*ns, 4, h, w)
        return ltrb
