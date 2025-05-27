import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            # embed_dim = None
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        # 单通道预测头（用于两个decoder输出）
        # self.pred_head_1 = nn.Conv2d(embed_dim, 1, kernel_size=1)
        # self.pred_head_2 = nn.Conv2d(embed_dim, 1, kernel_size=1)

        # # 可学习融合权重，初始化为0.5
        # self.alpha = nn.Parameter(torch.tensor(0.5))


    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def forward(self, x, use_gather_loss=True, return_patch_tokens=False):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        # Fused encoder patch tokens
        patch_tokens = self.fuse_feature(en_list)
        # x = self.fuse_feature(en_list)

        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), patch_tokens)
        if use_gather_loss:
            g_loss = self.gather_loss(patch_tokens, agg_prototype)
        else:
            g_loss = torch.tensor(0.0, device=patch_tokens.device)

        x = patch_tokens
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        # anomaly map from two decoder outputs
        # anomaly_map_1 = self.pred_head_1(de[0])  # shape: [B, 1, H, W]
        # anomaly_map_2 = self.pred_head_2(de[1])  # shape: [B, 1, H, W]

        # # 加权融合
        # anomaly_map = self.alpha * anomaly_map_1 + (1 - self.alpha) * anomaly_map_2
        # anomaly_map = torch.sigmoid(anomaly_map)  # 映射为概率图

        if return_patch_tokens:
            return en, de, g_loss, patch_tokens, agg_prototype
        return en, de, g_loss, agg_prototype
    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)









































