import torch
import torch.nn as nn

from typing import Tuple
from .modules import MLP, PositionEmbeddingRandom, DualMixAttentionBlock, LayerNorm2d

from configs.base import Config


class MaskDecoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.num_mask_tokens = cfg.num_classes
        self.mask_tokens = nn.Embedding(
            self.num_mask_tokens, cfg.encoder_out_features[-1]
        )
        self.indices = list(range(len(cfg.encoder_out_features)))[::-1]

        for block_index in self.indices:
            setattr(
                self,
                f"DMAB_s{block_index}",
                DualMixAttentionBlock(
                    depth=cfg.mask_depths[block_index],
                    embedding_dim=cfg.encoder_out_features[block_index],
                    mlp_dim=cfg.mask_mlp_dim,
                    num_heads=cfg.mask_num_head,
                ),
            )
            out_dim = (
                cfg.encoder_out_features[block_index - 1]
                if block_index > 0
                else cfg.encoder_out_features[block_index]
            )
            setattr(
                self,
                f"MLP_s{block_index}",
                nn.ModuleList(
                    [
                        MLP(
                            cfg.encoder_out_features[block_index],
                            cfg.encoder_out_features[block_index],
                            out_dim,
                            3,
                        )
                        for _ in range(self.num_mask_tokens)
                    ]
                ),
            )
            setattr(
                self,
                f"PE_s{block_index}",
                PositionEmbeddingRandom(cfg.encoder_out_features[block_index] // 2),
            )
            if block_index > 0:
                setattr(
                    self,
                    f"SUB_s{block_index}",
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            cfg.encoder_out_features[block_index],
                            cfg.encoder_out_features[block_index - 1],
                            kernel_size=2,
                            stride=2,
                        ),
                        LayerNorm2d(cfg.encoder_out_features[block_index - 1]),
                        nn.GELU(),
                    ),
                )
            else:
                setattr(
                    self,
                    f"SUB_s{block_index}",
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            cfg.encoder_out_features[block_index],
                            cfg.encoder_out_features[block_index] // 2,
                            kernel_size=2,
                            stride=2,
                        ),
                        LayerNorm2d(cfg.encoder_out_features[block_index] // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            cfg.encoder_out_features[block_index] // 2,
                            cfg.encoder_out_features[block_index],
                            kernel_size=2,
                            stride=2,
                        ),
                        LayerNorm2d(cfg.encoder_out_features[block_index]),
                        nn.GELU(),
                    ),
                )

    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder

        Returns:
          torch.Tensor: batched predicted masks
        """
        query_tokens = self.mask_tokens.weight.unsqueeze(0).expand(
            image_embeddings[-1].size(0), -1, -1
        )

        previous_feat = torch.zeros_like(image_embeddings[-1])
        current_feat = torch.zeros_like(image_embeddings[-1])
        for block_index in self.indices:
            feat = image_embeddings[block_index]
            # Expand per-image data in batch direction to be per-mask
            if feat.shape[0] != query_tokens.shape[0]:
                current_feat = torch.repeat_interleave(
                    feat, query_tokens.shape[0], dim=0
                )
            else:
                current_feat = feat
            current_feat = current_feat + previous_feat

            b, c, h, w = current_feat.shape

            pos_src = torch.repeat_interleave(
                getattr(self, f"PE_s{block_index}")(feat.shape[2:]).unsqueeze(0).cpu(),
                query_tokens.shape[0],
                dim=0,
            ).to(current_feat.device)

            att_tokens, current_feat = getattr(self, f"DMAB_s{block_index}")(
                query_tokens,
                current_feat,
                pos_src,
            )

            current_feat = previous_feat = getattr(self, f"SUB_s{block_index}")(
                current_feat
            )
            query_tokens = torch.stack(
                [
                    getattr(self, f"MLP_s{block_index}")[i](att_tokens[:, i, :])
                    for i in range(self.num_mask_tokens)
                ],
                dim=1,
            )

        b, c, h, w = current_feat.shape
        masks = (query_tokens @ current_feat.view(b, c, h * w)).view(b, -1, h, w)

        return masks, current_feat
