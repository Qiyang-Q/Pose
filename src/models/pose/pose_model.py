import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import src.models.ibrnet.mlp_network
from src.models.matchers.positional_encoding import PositionalEncoding, PositionEncodingSine, KeypointEncoding_linear
from src.utils.pose_utils import *
from src.models.ibrnet.mlp_network import DenseLayer
from src.utils.debug_utils import *
from src.models.matchers.transformer import LocalFeatureTransformer
from src.models.matchers.coarse_matching import CoarseMatching


class pose_model(nn.Module):
    def __init__(self, featnet, ibrnet, matcher, pnp_solver, use_weightmap=False, use_nerf_mlp=False, resolution=128):
        super().__init__()
        self.featnet = featnet
        self.ibrnet = ibrnet

        # Transformer I use
        self.matcher = matcher
        self.pnp_solver = pnp_solver
        self.bottleneck_layer = DenseLayer(self.featnet.out_channel * 2, self.featnet.out_channel)
        self.use_weightmap = use_weightmap
        self.use_nerf_mlp = use_nerf_mlp

        # My positional encoding
        # self.pos_encoder_2d = PositionalEncoding(dim=2)
        # self.pos_encoder_3d = PositionalEncoding(dim=3)

        # resolution to sample from feature map
        self.sample_res = int(resolution / 2)
        self.res = resolution

        # OnePose's positional encoding
        self.pos_encoder_2d = PositionEncodingSine(d_model=self.featnet.out_channel,
                                                   max_shape=(self.sample_res, self.sample_res))
        self.pos_encoder_3d = KeypointEncoding_linear(inp_dim=3,
                                                      feature_dim=self.featnet.out_channel,
                                                      layers=[32, 64, 128],
                                                      norm_method='instancenorm')

        # 2D to 3D correspondence sampling number used to feed into PnP solver
        self.crd_sample = 2000

        # Debug tools
        self.debug = False
        self.debug_dict = {}

        # From EProPnP, ok to skip this for now
        self.scale_branch = nn.Linear(self.featnet.out_channel, 2)
        self.weight_branch = nn.Conv2d(self.featnet.out_channel, 2, kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.weight_branch.weight, mean=0, std=0.001)
        nn.init.constant_(self.scale_branch.weight, 0)
        nn.init.constant_(self.scale_branch.bias, 3.0)
        self.weight_branch.weight.data[3:] = 0

        # Transformer used in OnePose
        self.loftr = LocalFeatureTransformer({'type': 'LoFTR', 'd_model': 128, 'nhead': 8, 'd_ffm': 128,
                                              'layer_names': ['self', 'cross'], 'layer_iter_n': 3, 'dropout': 0.,
                                              'attention': 'linear', 'norm_method': 'layernorm', 'kernel_fn': "elu + 1",
                                              'd_kernel': 16,
                                              'redraw_interval': 2,
                                              'rezero': None,
                                              'final_proj': False})

    def forward(self, data):

        # Extract 2D and 3D features
        feat_2d, enc_2d, weight_2d, scale = self.feat_extract_2d(data)
        feat_3d, enc_3d = self.feat_extract_3d(data)
        if self.debug:
            vis_aggre_feat(data, self.debug_dict, weight_2d)

        # Solve correspondence
        x2d, x3d, w2d, confidence, similarity = self.solve_correspondence(feat_2d, feat_3d, enc_2d, enc_3d, weight_2d, data)
        if self.debug:
            vis_attn_feat(data, self.debug_dict, self.featnet)

        # Solve pose
        pose = self.solve_pose(x2d, x3d, w2d, data)

        if self.training:
            return pose, similarity, confidence, scale
        else:
            return pose

    def feat_extract_2d(self, data):
        rgb = data['rgb_norm']
        x2d = data['pixel']

        # Extract 2D features from query image
        featmap = self.featnet(rgb.permute(0, 3, 1, 2))

        # Extract weightmap and scale from query feature map, used in EProPnP, can skip this for now
        weightmap = self.weight_branch(featmap).permute(0, 2, 3, 1)
        scale = self.scale_branch(featmap.flatten(2).mean(dim=-1)).exp()

        # grid sample query feature map
        x2d_norm = normalize(x2d, self.sample_res).unsqueeze(2)
        feat_2d = F.grid_sample(featmap, x2d_norm, align_corners=True, padding_mode='border').permute(0, 2, 3, 1).squeeze(2)

        # Positional encoding
        # feat_2d, enc_2d = self.pos_encoder_2d(feat_2d, norm_coor_2d(x2d, half_res))
        feat_2d, enc_2d = self.pos_encoder_2d(feat_2d)

        # Debug tools
        self.debug_dict['featmap'] = featmap if self.debug else None

        return feat_2d, enc_2d, weightmap, scale

    def feat_extract_3d(self, data):
        batch_size = data['voxel'].shape[0]
        voxel_samples = data['voxel'].shape[1]

        # Sample reference feature from reference images visible to sampled voxels
        ref_feat, ref_featmaps = sample_ref_feat(ref_rgbs=data['ref_rgbs'],
                                   inv_crd=data['inv_crd'],
                                   encoder=self.featnet,
                                   ref_intr=data['ref_intr'],
                                   ref_pose=data['ref_pose'],
                                   voxel=data['voxel'],
                                   use_nerf_mlp=self.use_nerf_mlp,
                                   res=self.res)

        # Not working yet, ignore this for now
        if self.use_nerf_mlp:
            ref_feat = ref_feat.view(-1, 1, ref_feat.shape[-2], ref_feat.shape[-1])
            voxel = data['voxel'].view(-1, 1, 3)
            visibility = data['visibility'].view(-1, 1, data['visibility'].shape[-1], 1)
            feat_3d = self.ibrnet(pos=voxel,
                                  dir=None,
                                  rgb_feat=ref_feat,
                                  mask=visibility,
                                  feat_only=True)
            feat_3d = self.bottleneck_layer(feat_3d)
            feat_3d = feat_3d.view(batch_size, voxel_samples, -1)
        else:

            # Transparency obtained from NeRF
            transparency = data['inv_crd'][..., 1].unsqueeze(-1)

            # Transparency weighted sum of features
            feat_3d = (ref_feat * transparency).sum(dim=2) / transparency.sum(dim=2)

        # Debug tools
        self.debug_dict['feat_3d'] = feat_3d if self.debug else None
        self.debug_dict['ref_featmaps'] = ref_featmaps if self.debug else None

        # Positional encoding
        feat_3d, enc_3d = self.pos_encoder_3d(kpts=data['voxel_norm'], descriptors=feat_3d)
        # feat_3d, enc_3d = self.pos_encoder_3d(feat_3d, norm_coor_3d(data['voxel_norm']))

        return feat_3d, enc_3d

    def solve_correspondence(self, feat_2d, feat_3d, enc_2d, enc_3d, weightmap, data):
        x2d = data['pixel']
        x3d = data['voxel']
        # similarity, layer_dict = self.matcher(feat_2d, feat_3d, enc_2d, enc_3d)
        # self.debug_dict['layer_dict'] = layer_dict if self.debug else None

        # Cross attention between 2D and 3D features
        feat_3d_attn, feat_2d_attn = self.loftr(feat_3d, feat_2d, query_mask=None)

        # Debug tools
        self.debug_dict['layer_dict'] = {0: [feat_3d_attn, feat_2d_attn]} if self.debug else None

        # Obtain similarity matrix
        similarity = torch.bmm(feat_3d_attn / math.sqrt(feat_3d_attn.shape[-1]), feat_2d_attn.transpose(-2, -1))

        # Parse similarity matrix to align 2D coordinates, 3D coordinates and their corresponding weights
        confidence, x3d, x2d, w2d = parse_similarity(similarity, x2d, x3d, weightmap)

        # Sample correspondence
        confidence, x3d, x2d, w2d = sample_similarity(confidence, x3d, x2d, w2d, strategy='prob', init_sample=self.crd_sample)

        # Flip x2d from (y, x) to (x, y) for PnP solver
        x2d = x2d.flip(dims=[-1])

        return x2d, x3d, w2d, confidence, similarity

    def solve_pose(self, x2d, x3d, w2d, data):
        intrinsic = data['query_intr'][:, :3, :3]
        pose_gt = data['query_pose_quat']
        if self.training:
            pose_pred, logweights, cost = self.pnp_solver(x3d, x2d, w2d, intrinsic, pose_gt)
            return pose_pred, logweights, cost
        else:
            pose_pred = self.pnp_solver(x3d, x2d, w2d, intrinsic, pose_gt=None)
            return pose_pred


