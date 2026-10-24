import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class WeatherEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class SoilEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class SeasonEncoder(nn.Module):
    def __init__(self, num_seasons=2, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_seasons, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_features, environmental_features):
        attended, attention_weights = self.attention(
            query=image_features,
            key=environmental_features,
            value=environmental_features,
            need_weights=True,
        )
        output = self.norm(image_features + attended)
        return output, attention_weights


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.transformer(x))


class FullMangoMultimodalModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.vit = vit_b_16(weights=None, num_classes=num_classes)

        self.weather_encoder = WeatherEncoder(3, 64, 256)
        self.soil_encoder = SoilEncoder(2, 64, 256)
        self.season_encoder = SeasonEncoder(2, 256)

        self.environment_projection = nn.Linear(256, 768)
        self.cross_attention = CrossAttention(768, 8)
        self.fusion_transformer = FusionTransformer(768, 8, 2)

        self.disease_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def extract_image_tokens(self, image):
        x = self.vit._process_input(image)
        batch_size = x.shape[0]

        cls_token = self.vit.class_token.expand(
            batch_size, -1, -1
        )
        x = torch.cat((cls_token, x), dim=1)
        return self.vit.encoder(x)

    def forward(
        self, image, weather, soil, season, return_attention=False
    ):
        image_tokens = self.extract_image_tokens(image)

        weather_features = self.weather_encoder(weather)
        soil_features = self.soil_encoder(soil)
        season_features = self.season_encoder(season)

        environment = (
            weather_features + soil_features + season_features
        )
        environment = self.environment_projection(environment)
        environment_token = environment.unsqueeze(1)

        attended_image, attention_weights = self.cross_attention(
            image_features=image_tokens,
            environmental_features=environment_token,
        )

        fused_features = self.fusion_transformer(attended_image)
        cls_features = fused_features[:, 0]
        logits = self.disease_head(cls_features)

        if return_attention:
            return {
                "logits": logits,
                "attention": attention_weights,
                "image_tokens": image_tokens,
                "environment_features": environment,
                "fused_features": fused_features,
                "cls_features": cls_features,
            }

        return logits
