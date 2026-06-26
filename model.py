import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, AutoProcessor

class LoRALinear(nn.Module):
    """
    Injects a Low-Rank Adaptation (LoRA) matrix side-by-side with a frozen linear layer.
    """
    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Define low-rank matrices
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Reset parameters following standard initialization conventions
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen mapping output
        base_output = self.original_linear(x)
        # Low rank update path calculation
        lora_output = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_output + lora_output * self.scaling


class HybridTemporalEventAdapter(nn.Module):
    """
    Combines local chronological mechanics (Conv1D) and global structural 
    dependencies (Multi-head Self-Attention) across the frame sequence.
    """
    def __init__(self, dim=768, num_heads=8, kernel_size=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # Local temporal branch (neighboring frame adjustments)
        self.local_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        # Global temporal branch (cross-sequence dependencies)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Learnable sequence fusion factors
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Feed-Forward network tracking block
        hidden = dim * mlp_ratio
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        z = self.norm1(x)

        # 1. Local branch routing
        local = z.transpose(1, 2)
        local = self.local_conv(local)
        local = local.transpose(1, 2)

        # 2. Global attention routing
        global_feat, _ = self.attn(z, z, z)

        # 3. Blending and residual aggregation
        h = x + self.alpha * local + self.beta * global_feat

        # 4. FFN normalization map
        y = h + self.ffn(self.norm2(h))
        return y


class Siglip2LoRATemporalBridge(nn.Module):
    """
    Level 4: Parameter-Efficient Temporal Alignment
    Vision: Frozen SigLIP + Trainable Spatial LoRA (Q, V) + Trainable Hybrid Temporal Event Adapter.
    Text: Frozen text encoder with a manual prompt.
    Purpose: Evaluates whether slight, parameter-efficient fine-tuning of spatial 
             representations alongside temporal sequencing boosts action classification.
    """
    def __init__(self, model_name="google/siglip2-base-patch16-224", class_names=None, lora_r=4, lora_alpha=8.0):
        super().__init__()
        if class_names is None:
            raise ValueError("You must provide a list of class_names.")
        
        self.class_names = class_names
        
        # 1. Load full SigLIP foundational framework
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 2. Extract configuration dimensions safely
        if hasattr(self.model.config, "text_config") and self.model.config.text_config is not None:
            if isinstance(self.model.config.text_config, dict):
                embedding_dim = self.model.config.text_config.get("hidden_size", 768)
            else:
                embedding_dim = getattr(self.model.config.text_config, "hidden_size", 768)
        else:
            embedding_dim = getattr(self.model.config, "hidden_size", 768)

        # 3. INJECT LoRA INTO VISION ENCODER SELF-ATTENTION
        self._apply_vision_lora(r=lora_r, alpha=lora_alpha)

        # 4. FREEZE BASE MODULES EXCEPT FOR GRADIENT TARGETS
        # LoRA parameters explicitly remain active because they are attached as fresh sub-modules
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # 5. INITIALIZE THE TRAINABLE TEMPORAL ADAPTER
        self.temporal_adapter = HybridTemporalEventAdapter(dim=embedding_dim)
        
        # Prompts structural setups
        self.prompts = [f"A video of {cls_name}" for cls_name in class_names]
        self._text_features = None 

    def _apply_vision_lora(self, r, alpha):
        """Finds and replaces multi-head self-attention query/value targets with LoRA layers."""
        num_injected_layers = 0
        for layer in self.model.vision_model.encoder.layers:
            if hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn, "v_proj"):
                layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r, alpha=alpha)
                layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r, alpha=alpha)
                num_injected_layers += 2
        print(f"Successfully injected {num_injected_layers} spatial LoRA parameter layers into Vision Backbone.")

    def train(self, mode: bool = True):
        """Custom train mode enforcement to guarantee baseline isolation."""
        super().train(mode)
        if mode:
            # Keep underlying structural weights (like non-LoRA layers and LayerNorms) locked
            self.model.eval()
            self.temporal_adapter.train()
            
            # Keep only the custom spatial LoRA adapter layers active in training mode
            for m in self.model.modules():
                if isinstance(m, LoRALinear):
                    m.train()
        return self

    def _get_text_features(self, device):
        if self._text_features is not None:
            return self._text_features
        
        inputs = self.processor.tokenizer(text=self.prompts, padding="longest", truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        text_features = F.normalize(text_features, p=2, dim=-1)
        self._text_features = text_features
        return self._text_features

    def forward(self, pixel_values):
        # pixel_values shape: (B, T, C, H, W)
        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # 1. Flatten B and T to stream spatial elements through the updated 2D vision encoder
        pixel_values = pixel_values.view(B * T, C, H, W)
        
        # 2. Extract spatial layout features (Gradients dynamically backpropagate through the spatial LoRA weights)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        spatial_sequence = vision_outputs.last_hidden_state  # (B*T, S, D)
        
        # 3. Spatial Pooling: Compress per-frame patches into feature representations
        frame_features = self.model.vision_model.head(spatial_sequence)  # (B*T, D)
        frame_features = frame_features.view(B, T, -1)  # (B, T, D)
        
        # 4. Pass chronological records through the Trainable Temporal Adapter
        video_features = self.temporal_adapter(frame_features)  # (B, T, D)
        
        # 5. Average temporal records across time sequence blocks
        video_features = video_features.mean(dim=1)  # (B, D)
        
        # 6. L2 Normalize video representations
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        # 7. Fetch normalized completely frozen text prompt embeddings
        text_features = self._get_text_features(device)  # (Num_Classes, D)
        
        # 8. Compute contrastive logits output using the SigLIP exponential scalar multiplier
        logit_scale = self.model.logit_scale.exp()
        logits = (video_features @ text_features.T) * logit_scale  # (B, Num_Classes)
        
        return logits