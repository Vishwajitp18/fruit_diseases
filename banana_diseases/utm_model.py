"""UTM-AI starter model: ViT image encoder + optional environment cross-attention."""

from typing import Dict, Optional
import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class UTMModel(nn.Module):
    """Multimodal Vision Transformer model with optional environmental sequence cross-attention."""

    def __init__(
        self,
        num_classes: int,
        env_dim: int = 1,
        use_environment: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()
        self.use_environment = bool(use_environment)
        self.env_dim = int(env_dim)
        self.last_visual_tokens = None
        self.last_cross_attention = None

        if self.use_environment:
            self.env_projection = nn.Sequential(
                nn.Linear(self.env_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
            )
            self.env_lstm = nn.LSTM(
                input_size=256,
                hidden_size=384,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )  # 384 * 2 = 768, matching ViT-B/16 tokens
            self.env_norm = nn.LayerNorm(768)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=768,
                num_heads=8,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(768)
            self.fusion = nn.Sequential(
                nn.Linear(768 * 2, 768),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.LayerNorm(768),
            )

        self.disease_head = nn.Linear(768, num_classes)
        self.severity_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: torch.Tensor,
        env_sequence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # torchvision ViT representation: [CLS] + 196 image patch tokens for 224x224 inputs.
        patches = self.vit._process_input(images)
        cls = self.vit.class_token.expand(patches.shape[0], -1, -1)
        visual_tokens = self.vit.encoder(torch.cat([cls, patches], dim=1))
        self.last_visual_tokens = visual_tokens
        self.last_cross_attention = None
        if torch.is_grad_enabled() and visual_tokens.requires_grad:
            visual_tokens.retain_grad()

        visual_summary = visual_tokens[:, 0, :]
        if self.use_environment:
            if env_sequence is None:
                raise ValueError("This checkpoint requires an 8-step environmental sequence.")
            env_embedded = self.env_projection(env_sequence)
            env_tokens, _ = self.env_lstm(env_embedded)
            env_tokens = self.env_norm(env_tokens)
            cross_features, attn = self.cross_attention(
                query=env_tokens,
                key=visual_tokens,
                value=visual_tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            self.last_cross_attention = attn.detach()
            cross_features = self.cross_norm(cross_features + env_tokens)
            fused = self.fusion(
                torch.cat([visual_summary, cross_features.mean(dim=1)], dim=1)
            )
        else:
            fused = visual_summary

        return {
            "disease_logits": self.disease_head(fused),
            "severity": self.severity_head(fused).squeeze(-1),
            "risk": self.risk_head(fused).squeeze(-1),
        }

