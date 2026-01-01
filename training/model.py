"""
MobileNetV3 + ArcFace model for MTG card recognition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin) classification head.

    Adds angular margin penalty to force better class separation in embedding space.
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        # Learnable class centers (weights)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Normalized embeddings (B, embedding_dim)
            labels: Class labels (B,)

        Returns:
            Logits with angular margin applied (B, num_classes)
        """
        # Normalize weights
        weight_norm = F.normalize(self.weight, dim=1)

        # Cosine similarity (since embeddings are normalized)
        cosine = F.linear(embeddings, weight_norm)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # Calculate sin from cos
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle edge case where cos(theta) < cos(pi - m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to correct class
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine

        # Scale logits
        output = output * self.scale

        return output


class CardEmbeddingModel(nn.Module):
    """
    MobileNetV3-based embedding model for card recognition.

    Architecture:
    - MobileNetV3-Large backbone (pretrained)
    - Global average pooling
    - Embedding layer (512-dim, L2 normalized)
    - ArcFace head (for training)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load MobileNetV3-Large backbone
        self.backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )

        # Get backbone output dimension
        backbone_out_dim = self.backbone.classifier[0].in_features

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # ArcFace classification head (only used during training)
        self.arcface = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=30.0,
            margin=0.5,
        )

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embedding from image.

        Args:
            x: Input image tensor (B, 3, 224, 224)

        Returns:
            L2-normalized embedding (B, embedding_dim)
        """
        features = self.backbone(x)
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        During training (labels provided): Returns ArcFace logits
        During inference (no labels): Returns normalized embeddings
        """
        embedding = self.get_embedding(x)

        if labels is not None:
            # Training mode - return logits
            logits = self.arcface(embedding, labels)
            return logits
        else:
            # Inference mode - return embeddings
            return embedding


class CardRecognizer:
    """
    High-level interface for card recognition.

    Usage:
        recognizer = CardRecognizer("model.onnx", "embeddings.faiss", "cards.json")
        card_name, confidence = recognizer.identify(card_image)
    """

    def __init__(
        self,
        model_path: str,
        embeddings_path: str,
        cards_path: str,
        device: str = "cpu",
    ):
        # This will be implemented after training
        raise NotImplementedError("CardRecognizer will be implemented after training")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    model = CardEmbeddingModel(num_classes=1000, embedding_dim=512)
    print(f"Model created with {count_parameters(model):,} parameters")

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 1000, (4,))

    # Training mode
    logits = model(x, labels)
    print(f"Training output shape: {logits.shape}")

    # Inference mode
    embeddings = model(x)
    print(f"Inference output shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings, dim=1)}")  # Should be ~1.0
