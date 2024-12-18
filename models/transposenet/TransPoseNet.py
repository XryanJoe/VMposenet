"""
The TransPoseNet model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer_encoder import Transformer
from .pencoder import  nested_tensor_from_tensor_list
from .backbone import build_backbone
from .vision_mamba import Vim


class TransPoseNet(nn.Module):

    def __init__(self, config, pretrained_path):
        """
        config: (dict) configuration of the model
        pretrained_path: (str) path to the pretrained backbone
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = True

        # CNN backbone
        self.backbone = build_backbone(config)

        # Position (t) and orientation (rot) encoders
        self.vim_t = Vim(config)
        self.vim_rot = Vim(config)

        dim = self.vim_t.dim

        # The learned pose token for position (t) and orientation (rot)
        self.pose_token_embed_t = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)
        self.pose_token_embed_rot = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)

        self.num_scenes = config.get("num_scenes")
        self.multiscene = False
        self.classify_scene = config.get("classify_scene")
        if self.num_scenes is not None and self.num_scenes > 1:
            self.scene_embed = nn.Linear(1, dim)
            self.multiscene = True
            if self.classify_scene:
                self.avg_pooling = nn.AdaptiveAvgPool2d(1)
                self.scene_cls = nn.Sequential(nn.Linear(1280 , self.num_scenes),
                                               nn.LogSoftmax(dim=1))

        # The projection of the activation map before going into the Transformer's encoder
        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], dim, kernel_size=1)

        # Whether to use prior from the position for the orientation
        self.use_prior = config.get("use_prior_t_for_rot")

        # Regressors for position (t) and orientation (rot)
        self.regressor_head_t = PoseRegressor(dim, 3)
        self.regressor_head_rot = PoseRegressor(dim, 4, self.use_prior)

    def forward_transformers(self, data):
        """
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return a dictionary with the following keys--values:
            global_desc_t: latent representation from the position encoder
            global_dec_rot: latent representation from the orientation encoder
        """
        samples = data.get('img')

        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # Extract the features and the position embedding from the visual backbone
        features, pos = self.backbone(samples)

        # src_t, mask_t = features[0].decompose()
        # src_rot, mask_rot = features[1].decompose()
        # print(self.input_proj_t(src_t).shape,"--------------------")
        src_t=features[0]
        src_rot=features[1]

        # Run through the transformer to translate to "camera-pose" language
        # assert mask_t is not None
        # assert mask_rot is not None

        bs = src_t.shape[0]
        pose_token_embed_rot = self.pose_token_embed_rot.unsqueeze(1).repeat(1, bs, 1)
        pose_token_embed_t = self.pose_token_embed_t.unsqueeze(1).repeat(1, bs, 1)

        scene_dist = None
        # if self.multiscene:
        #     selected_scene = data.get("scene")
        #     if self.classify_scene:
        #         src_scene, _ = features[2]()
        #         src_scene = self.avg_pooling(src_scene).flatten(1)
        #         scene_dist = self.scene_cls(src_scene)
        #     if selected_scene is None: # test time
        #         assert(self.classify_scene)
        #         selected_scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32)
        #     else:
        #         selected_scene = selected_scene.unsqueeze(1)

        #     scene_embed = self.scene_embed(selected_scene)
        #     pose_token_embed_rot = scene_embed + pose_token_embed_rot
        #     pose_token_embed_t = scene_embed + pose_token_embed_t

        local_descs_t = self.vim_t(self.input_proj_t(src_t), pos[0], pose_token_embed_t)
        local_descs_rot = self.vim_rot(self.input_proj_rot(src_rot), pos[1],
                                               pose_token_embed_rot)

        # Take the global desc from the pose token
        global_desc_t = local_descs_t[:, 0, :]
        global_desc_rot = local_descs_rot[:, 0, :]

        return {'global_desc_t':global_desc_t, 'global_desc_rot':global_desc_rot, "scene_dist":scene_dist}

    def forward_heads(self, transformers_res):
        """
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        returns: dictionary with key-value 'pose'--expected pose (NX7)
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')

        x_t = self.regressor_head_t(global_desc_t)
        if self.use_prior:
            global_desc_rot = torch.cat((global_desc_t, global_desc_rot), dim=1)

        x_rot = self.regressor_head_rot(global_desc_rot)
        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return {'pose': expected_pose, "scene_dist":transformers_res.get("scene_dist")}

    def forward(self, data):
        """ The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED

            returns dictionary with key-value 'pose'--expected pose (NX7)
        """
        transformers_encoders_res = self.forward_transformers(data)
        # Regress the pose from the image descriptors
        heads_res = self.forward_heads(transformers_encoders_res)
        return heads_res


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)