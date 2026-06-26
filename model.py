# Level 3: The Temporal Bridge (Frozen Spatial Backbone + Trainable Local-Global Temporal Adapter)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, AutoProcessor

class HybridTemporalEventAdapter(nn.Module):
    """
    Combines local chronological mechanics (Conv1D) and global structural 
    dependencies (Multi-head Self-Attention) across the frame sequence.
    """
    def __init__(self, dim=768, num_heads=8, kernel_size=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # Local temporal branch (captures neighboring frame variations)
        self.local_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        # Global temporal branch (captures non-adjacent sequential dependencies)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Learnable sequence fusion parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Feed-Forward Network layer
        hidden = dim * mlp_ratio
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        # x shape: (B, T, D)
        z = self.norm1(x)

        # 1. Process local dynamics
        local = z.transpose(1, 2)
        local = self.local_conv(local)
        local = local.transpose(1, 2)

        # 2. Process global dynamics
        global_feat, _ = self.attn(z, z, z)

        # 3. Residual Feature Fusion
        h = x + self.alpha * local + self.beta * global_feat

        # 4. FFN Processing block
        y = h + self.ffn(self.norm2(h))
        return y


class Siglip2TemporalBridgeBaseline(nn.Module):
    """
    Level 3: The Temporal Bridge
    Vision: Frozen SigLIP + Trainable Hybrid Temporal Event Adapter.
    Text: Frozen text encoder with a rigid, manual prompt ("A video of {class}").
    Purpose: Tracks chronological mechanics (like patterns over time) by routing 
             frame sequences through local and global temporal branches before final matching.
    """
    def __init__(self, model_name="google/siglip2-base-patch16-224", class_names=None):
        super().__init__()
        if class_names is None:
            raise ValueError("You must provide a list of class_names.")
        
        self.class_names = class_names
        
        # 1. Load full SigLIP model foundation
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 2. FREEZE the entire foundational base model (Both Vision and Text towers)
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
        # 3. Extract embedding dimension safely to eliminate Pylance attribute access errors
        if hasattr(self.model.config, "text_config") and self.model.config.text_config is not None:
            if isinstance(self.model.config.text_config, dict):
                embedding_dim = self.model.config.text_config.get("hidden_size", 768)
            else:
                embedding_dim = getattr(self.model.config.text_config, "hidden_size", 768)
        else:
            embedding_dim = getattr(self.model.config, "hidden_size", 768)
        
        # 4. THE ONLY TRAINABLE BLOCK FOR LEVEL 3: Explicitly isolated for full fine-tuning
        self.temporal_adapter = HybridTemporalEventAdapter(dim=embedding_dim)
        
        # Pre-prepare rigid text prompts and cache placeholder structures
        self.prompts = [f"A video of {cls_name}" for cls_name in class_names]
        self._text_features = None 
    
    def train(self, mode: bool = True):
        """Custom train mode enforcement to guarantee baseline isolation."""
        super().train(mode)
        if mode:
            # Enforce that the heavy foundation backbone always behaves under eval constraints
            # This locks LayerNorm states and turns off dropout in SigLIP
            self.model.eval()
            # Explicitly keep the baseline adapter active and training
            self.temporal_adapter.train()
        return self

    def _get_text_features(self, device):
        """Encodes the manual prompts once and caches them securely (remains completely frozen)."""
        if self._text_features is not None:
            return self._text_features
        
        inputs = self.processor.tokenizer(
            text=self.prompts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # (Num_Classes, D)
        
        # Normalize text features
        text_features = F.normalize(text_features, p=2, dim=-1)
        self._text_features = text_features
        return self._text_features

    def forward(self, pixel_values):
        # pixel_values shape: (B, T, C, H, W)
        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # 1. Flatten B and T to pass through the 2D vision encoder
        pixel_values = pixel_values.view(B * T, C, H, W)
        
        # 2. Extract spatial sequence from frozen backbone
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        spatial_sequence = vision_outputs.last_hidden_state  # (B*T, S, D)
        
        # 3. Pool spatially PER FRAME using SigLIP's attention head
        # This converts each frame into a single descriptive 768-dim token
        frame_features = self.model.vision_model.head(spatial_sequence)  # (B*T, D)
        
        # 4. Unflatten time dimension to feed the temporal layout
        frame_features = frame_features.view(B, T, -1)  # (B, T, D)
        
        # 5. PASS THROUGH THE TRAINABLE ADAPTER (Level 3 fine-tuning)
        # Learns cross-frame structural dependencies (chronological order)
        video_features = self.temporal_adapter(frame_features)  # (B, T, D)
        
        # 6. Aggregate temporal features into a single video vector via mean pooling
        video_features = video_features.mean(dim=1)  # (B, D)
        
        # 7. L2 Normalize video features
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        # 8. Fetch normalized frozen text features
        text_features = self._get_text_features(device)  # (Num_Classes, D)
        
        # 9. Compute Logits using SigLIP's internal temperature scale
        logit_scale = self.model.logit_scale.exp()
        logits = (video_features @ text_features.T) * logit_scale  # (B, Num_Classes)
        
        return logits