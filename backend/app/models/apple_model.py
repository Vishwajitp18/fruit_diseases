import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class WeatherEncoder(nn.Module):
    def __init__(self, in_dim=5, out_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SoilEncoder(nn.Module):
    def __init__(self, in_dim=9, out_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LocationSeasonEncoder(nn.Module):
    def __init__(self, loc_dim=2, num_seasons=4, season_emb=16, out_dim=384):
        super().__init__()
        self.season_embed = nn.Embedding(num_seasons, season_emb)
        self.net = nn.Sequential(
            nn.Linear(loc_dim + season_emb, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, loc, season_idx):
        s = self.season_embed(season_idx)
        return self.net(torch.cat([loc, s], dim=1))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens, kv_tokens):
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        attn_out, attn_w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        x = q_tokens + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x, attn_w


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        num_classes=4,
        fusion_dim=384,
        num_heads=6,
        dropout=0.1,
        weather_dim=5,
        soil_dim=9,
        loc_dim=2,
        pretrained=False,
    ):
        super().__init__()
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
        vit_dim = self.vit.embed_dim
        self.vit_proj = (
            nn.Linear(vit_dim, fusion_dim) if vit_dim != fusion_dim else nn.Identity()
        )

        self.weather_enc = WeatherEncoder(weather_dim, fusion_dim)
        self.soil_enc = SoilEncoder(soil_dim, fusion_dim)
        self.loc_season_enc = LocationSeasonEncoder(loc_dim, 4, 16, fusion_dim)

        self.img_to_env = CrossAttentionBlock(fusion_dim, num_heads, dropout)

        self.pool_norm = nn.LayerNorm(fusion_dim)
        self.disease_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.early_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.last_attn_weights = None
        self.last_vit_hw = None

    def forward(self, img, weather, soil, loc, season_idx):
        vit_tokens_all = self.vit.forward_features(img)
        img_tokens = self.vit_proj(vit_tokens_all[:, 1:])
        B, N, D = img_tokens.shape

        weather_tok = self.weather_enc(weather).unsqueeze(1)
        soil_tok = self.soil_enc(soil).unsqueeze(1)
        loc_tok = self.loc_season_enc(loc, season_idx).unsqueeze(1)
        env_tokens = torch.cat([weather_tok, soil_tok, loc_tok], dim=1)

        fused_img, attn_w = self.img_to_env(img_tokens, env_tokens)
        self.last_attn_weights = attn_w.detach()
        self.last_vit_hw = (int(N**0.5), int(N**0.5))

        pooled = self.pool_norm(fused_img.mean(dim=1))

        disease_logits = self.disease_head(pooled)
        severity = torch.sigmoid(self.severity_head(pooled)).squeeze(1)
        early_logit = self.early_head(pooled).squeeze(1)
        confidence = F.softmax(disease_logits, dim=1).max(dim=1).values

        return {
            "disease_logits": disease_logits,
            "severity": severity,
            "early_logit": early_logit,
            "confidence": confidence,
            "pooled_features": pooled,
        }

