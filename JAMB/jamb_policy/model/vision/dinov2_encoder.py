"""
DinoV2 Vision Encoder
Wraps pretrained DinoV2 (with register tokens) for feature extraction from RGB images.

Patch size is 14, matching Pi3's patch size: feeding both encoders the same
238x322 resize yields an identical 17x23 patch grid, which keeps the two
token sets spatially aligned (needed for the planned 4D RoPE).
"""
import torch
import torch.nn as nn
from typing import Tuple


class DINOV2(nn.Module):
    """
    DINOV2 Vision Encoder
    Extracts patch tokens from RGB images using pretrained DinoV2.
    Register tokens are excluded from the returned patch tokens.
    No projection - uses raw DinoV2 features.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14_reg",  # dinov2_vitl14_reg, dinov2_vitb14_reg, etc.
        freeze: bool = True,
        repo_dir: str = "thirdparty/dinov2",  # Local repo directory
        weights_path: str = "pretrained/dinov2_vitl14_reg4_pretrain.pth",  # Local weights
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze

        # Load pretrained DinoV2 model from local directory.
        # The hub loader accepts a local filesystem path for `weights`.
        print(f"Loading {model_name} from local directory...")
        print(f"  Repo dir: {repo_dir}")
        print(f"  Weights: {weights_path}")

        self.dinov2 = torch.hub.load(
            repo_dir,
            model_name,
            source='local',
            pretrained=True,
            weights=weights_path,
        )

        # Get model specs
        self.patch_size = self.dinov2.patch_size  # 14
        self.embed_dim = self.dinov2.embed_dim  # 1024 for vitl, 768 for vitb
        self.output_dim = self.embed_dim  # No projection

        # Freeze backbone if specified
        if self.freeze:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
            print(f"  DinoV2 backbone frozen")
        else:
            print(f"  DinoV2 backbone trainable")

        print(f"  Model: {model_name}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Register tokens: {getattr(self.dinov2, 'num_register_tokens', 0)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DinoV2 features from images

        Args:
            images: [B, 3, H, W] RGB images already normalized with ImageNet stats.
                    H and W must be multiples of the patch size (14).

        Returns:
            features: [B, N_patches, embed_dim] patch features
                      (CLS and register tokens excluded)
        """
        B, C, H, W = images.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H}x{W}) must be divisible by patch size {self.patch_size}"

        if self.freeze:
            self.dinov2.eval()

        with torch.set_grad_enabled(not self.freeze):
            features = self.dinov2.forward_features(images)
            return features['x_norm_patchtokens']

    def get_num_patches(self, image_size: Tuple[int, int]) -> int:
        """
        Calculate number of patches for given image size
        Args:
            image_size: (H, W)
        Returns:
            num_patches: number of patches
        """
        H, W = image_size
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        return num_patches
