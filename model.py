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

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Re-initialize low-rank pathway weight distributions
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_linear(x)
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

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        hidden = dim * mlp_ratio
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        z = self.norm1(x)
        
        local = z.transpose(1, 2)
        local = self.local_conv(local)
        local = local.transpose(1, 2)

        global_feat, _ = self.attn(z, z, z)
        h = x + self.alpha * local + self.beta * global_feat
        y = h + self.ffn(self.norm2(h))
        return y

class Siglip2FullLoRATemporalBridge(nn.Module):
    """
    Configurable Co-Alignment Parameter-Efficient Fine-Tuning Bridge.
    Allows toggling Vision LoRA, Text LoRA, and the Hybrid Temporal Adapter module.
    Enhanced with Dynamic Text Prompt Ensembles.
    """
    def __init__(
        self, 
        model_name="google/siglip2-base-patch16-224", 
        class_names=None, 
        lora_r=4, 
        lora_alpha=8.0,
        use_vision_lora=True,
        use_text_lora=True,
        use_temporal_adapter=True
    ):
        super().__init__()
        if class_names is None:
            raise ValueError("You must provide a list of class_names.")
        
        self.class_names = class_names
        
        self.use_vision_lora = use_vision_lora
        self.use_text_lora = use_text_lora
        self.use_temporal_adapter = use_temporal_adapter
        
        # 1. Load full SigLIP model foundation layers
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

        # 3. Conditionally inject LoRA into Vision Encoder
        if self.use_vision_lora:
            self._apply_vision_lora(r=lora_r, alpha=lora_alpha)
        else:
            print("Vision Backbone: LoRA skipped. Remaining frozen.")

        # 4. Conditionally inject LoRA into Text Encoder
        if self.use_text_lora:
            self._apply_text_lora(r=lora_r, alpha=lora_alpha)
        else:
            print("Text Tower: LoRA skipped. Remaining frozen.")

        # 5. Freeze foundational layers except for parameters containing "lora_"
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # 6. Conditionally initialize the Trainable Temporal Adapter
        if self.use_temporal_adapter:
            self.temporal_adapter = HybridTemporalEventAdapter(dim=embedding_dim)
            print("Temporal Block: HybridTemporalEventAdapter active.")
        else:
            self.temporal_adapter = None
            print("Temporal Block: Switched to basic mean pooling across the sequence.")
        
        # --- ITEM 1: Text Prompt Ensembling Templates ---
        self.templates = [
            "A video of {}",
            "A clip showing someone performing {}",
            "An action of {}",
            "A demonstration of {}",
            "A video recording of {}",
            "A video containing the action of {}",
            "A prompt showing a person doing {}"
        ]
        
        # Cache for precomputed text features when text tower is frozen
        self._cached_text_features = None

    def _apply_vision_lora(self, r, alpha):
        """Replaces Multi-Head Attention query/value weights in the Vision Transformer with LoRA layers."""
        num_injected = 0
        for layer in self.model.vision_model.encoder.layers:
            if hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn, "v_proj"):
                layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r, alpha=alpha)
                layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r, alpha=alpha)
                num_injected += 2
        print(f"Vision Backbone: Injected {num_injected} spatial LoRA projection layers.")

    def _apply_text_lora(self, r, alpha):
        """Replaces Multi-Head Attention query/value weights in the Text Encoder with LoRA layers."""
        num_injected = 0
        for layer in self.model.text_model.encoder.layers:
            if hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn, "v_proj"):
                layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r, alpha=alpha)
                layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r, alpha=alpha)
                num_injected += 2
        print(f"Text Tower: Injected {num_injected} linguistic LoRA projection layers.")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Crucial: Reset eval cache so training updates text LoRA weights dynamically
            self._cached_text_features = None 
            
            # Force foundational parameters into evaluation constraints
            self.model.eval()
            
            if self.temporal_adapter is not None:
                self.temporal_adapter.train()
            
            # Explicitly turn on training mode only for custom LoRALinear layers
            for m in self.model.modules():
                if isinstance(m, LoRALinear):
                    m.train()
        return self
        
    def _get_text_features(self, device):
        """Generates text features dynamically or returns cached versions during evaluation."""
        
        # 1. EVAlUATION CACHING: Always return cache if available during eval mode
        if not self.training and self._cached_text_features is not None:
            return self._cached_text_features.to(device)
            
        # 2. STATIC TRAINING CACHING: If training but Text LoRA is frozen, we can also use cache
        if self.training and not self.use_text_lora and self._cached_text_features is not None:
            return self._cached_text_features.to(device)

        # 3. COMPUTE EMBEDDINGS (Runs once for eval cache, or every batch if text LoRA is training)
        stacked_embeddings = []
        
        # Wrap the entire generation context under no_grad if gradients aren't needed
        context = torch.no_grad() if (not self.training or not self.use_text_lora) else contextlib.nullcontext()
        
        with context:
            for template in self.templates:
                prompts = [template.format(cls_name) for cls_name in self.class_names]
                inputs = self.processor.tokenizer(text=prompts, padding="longest", truncation=True, return_tensors="pt")
                
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                norm_features = F.normalize(text_outputs.pooler_output, p=2, dim=-1)
                stacked_embeddings.append(norm_features)
                
            mean_features = torch.stack(stacked_embeddings, dim=0).mean(dim=0)
            text_features = F.normalize(mean_features, p=2, dim=-1)

        # Cache the result if we are in validation/evaluation mode
        if not self.training:
            self._cached_text_features = text_features.cpu() # Store on CPU to save VRAM between epochs
            return self._cached_text_features.to(device)
            
        return text_features

    def forward(self, pixel_values):
        # pixel_values shape: (B, T, C, H, W)
        B, T, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # 1. Flatten B and T to map sequential frames through the 2D vision encoder
        pixel_values = pixel_values.view(B * T, C, H, W)
        
        # 2. Extract spatial layout features (Vision LoRA handles internal gradients)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        spatial_sequence = vision_outputs.last_hidden_state  # (B*T, S, D)
        
        # 3. Spatial Pooling
        frame_features = self.model.vision_model.head(spatial_sequence)  # (B*T, D)
        frame_features = frame_features.view(B, T, -1)  # (B, T, D)
        
        # 4. Conditionally process sequence with the Temporal Adapter or perform baseline mean pooling
        if self.use_temporal_adapter and self.temporal_adapter is not None:
            video_features = self.temporal_adapter(frame_features)  # (B, T, D)
            video_features = video_features.mean(dim=1)  # (B, D)
        else:
            video_features = frame_features.mean(dim=1)  # (B, D) directly maps to mean pooling

        # 5. L2 Normalize video representations
        video_features = F.normalize(video_features, p=2, dim=-1)
        
        # 6. Dynamically generate Text features containing Text LoRA gradients (if active)
        text_features = self._get_text_features(device)  # (Num_Classes, D)
        
        # 7. Compute contrastive logits output using the SigLIP exponential scalar multiplier
        logit_scale = self.model.logit_scale.exp()
        logits = (video_features @ text_features.T) * logit_scale  # (B, Num_Classes)
        
        return logits