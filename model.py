# action-siglip/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, AutoProcessor
from peft import LoraConfig
import re


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.original_linear = original_linear
        # Freeze base linear layer explicitly
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_linear(x)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_output


class HybridTemporalModule(nn.Module):
    def __init__(self, dim=768, num_heads=8, kernel_size=3):
        super().__init__()
        # 1. Local Temporal Convolution
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=kernel_size // 2)
        
        # 2. Global Temporal Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 3. Zero-initialized projection layer
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x shape: (B, T, C)
        residual = x
        
        # Local motion aggregation
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(residual + F.gelu(x_conv))
        
        # Global motion aggregation
        x_attn, _ = self.attn(x, x, x)
        x = self.norm2(x + x_attn)
        
        # Project and Temporal Mean Pool
        temporal_features = self.proj(x) 
        v = temporal_features.mean(dim=1)
        
        return v

class MetaNet(nn.Module):
    def __init__(self, dim, hidden_dim=512, num_prompt_tokens=4):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_prompt_tokens * dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v):
        B = v.size(0)
        delta_v = self.net(v)
        return delta_v.view(B, self.num_prompt_tokens, self.dim)


class Siglip2ActionModel(nn.Module):
    def __init__(
        self, 
        model_name: str, 
        class_names: list, 
        prompt_type: str,
        manual_prompt_template: str,
        cocoop_hidden_dim: int,
        lora_config: LoraConfig,
        unfreeze_backbone: bool = False
    ):
        super().__init__()
        if class_names is None:
            raise ValueError("class_names list required.")
        self.class_names = class_names
        self.prompt_type = prompt_type
        self.manual_prompt_template = manual_prompt_template
        
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if not unfreeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Backbone SigLIP is FROZEN.")
        else:
            print("Backbone SigLIP is UNFROZEN for Full Fine-Tuning!")

        text_config = getattr(self.model.config, "text_config", None)
        if isinstance(text_config, dict):
            embedding_dim = text_config.get("hidden_size", 768)
        elif text_config is not None:
            embedding_dim = getattr(text_config, "hidden_size", 768)
        else:
            embedding_dim = getattr(self.model.config, "hidden_size", 768)

        lora_r = getattr(lora_config, "r", 4)
        lora_alpha = getattr(lora_config, "lora_alpha", 8.0)
        # 1. Đọc danh sách target modules từ config
        lora_targets = getattr(lora_config, "target_modules", []) 

        # 2. Kiểm tra: Chỉ tiêm LoRA nếu r > 0 VÀ danh sách targets không rỗng
        if lora_r > 0 and len(lora_targets) > 0:
            print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}) to {lora_targets}...")
            # Truyền thêm tham số target_modules vào hàm
            self._apply_lora_to_encoder(self.model.vision_model.encoder, r=lora_r, alpha=lora_alpha, name="Vision", target_modules=lora_targets)
            # self._apply_lora_to_encoder(self.model.text_model.encoder, r=lora_r, alpha=lora_alpha, name="Text", target_modules=lora_targets)
        else:
            print("LoRA is disabled (r=0 or empty targets). Base model is fully frozen.")

        self.temporal_module = HybridTemporalModule(dim=embedding_dim)
        
        # --- FIX 2: THÊM LEARNABLE GATE GAMMA ĐỂ BẢO TỒN KHÔNG GIAN ZSL ---
        # Khởi tạo gamma = 0 để ban đầu mô hình dùng 100% feature chuẩn của SigLIP 2
        self.gamma = nn.Parameter(torch.zeros(1))
        
        if self.prompt_type == "cocoop":
            self.meta_net = MetaNet(dim=embedding_dim, hidden_dim=cocoop_hidden_dim, num_prompt_tokens=4)
        elif self.prompt_type == "manual":
            self.meta_net = None
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def _apply_lora_to_encoder(self, encoder, r, alpha, name, target_modules):
        num_injected = 0
        for layer in encoder.layers:
            attn = layer.self_attn
            
            # CHỈ tiêm nếu config có yêu cầu "q_proj"
            if hasattr(attn, "q_proj") and "q_proj" in target_modules:
                attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha)
                num_injected += 1
                
            # CHỈ tiêm nếu config có yêu cầu "k_proj"
            if hasattr(attn, "k_proj") and "k_proj" in target_modules:
                attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha)
                num_injected += 1
                
            # CHỈ tiêm nếu config có yêu cầu "v_proj"
            if hasattr(attn, "v_proj") and "v_proj" in target_modules:
                attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha)
                num_injected += 1
                
            # CHỈ tiêm nếu config có yêu cầu "out_proj"
            if hasattr(attn, "out_proj") and "out_proj" in target_modules:
                attn.out_proj = LoRALinear(attn.out_proj, r=r, alpha=alpha)
                num_injected += 1
                
        print(f"[{name} Encoder] {num_injected} LoRA matrices injected.")

    def forward(self, pixel_values, unseen_class_names=None, is_zero_shot=False, return_features=False):
        B, T, C_img, H, W = pixel_values.shape
        device = pixel_values.device
        
        # --- 1. VISION ENCODER ---
        pixel_values = pixel_values.view(B * T, C_img, H, W)
        image_embeds = self.model.get_image_features(pixel_values=pixel_values)

        if not isinstance(image_embeds, torch.Tensor):
            if hasattr(image_embeds, "pooler_output") and image_embeds.pooler_output is not None:
                image_embeds = image_embeds.pooler_output
            elif hasattr(image_embeds, "image_embeds"):
                image_embeds = image_embeds.image_embeds
            else:
                image_embeds = image_embeds[0] 
        
        spatial_features = image_embeds.view(B, T, -1)
        spatial_features_norm = F.normalize(spatial_features, p=2, dim=-1)

        base_v = spatial_features_norm.mean(dim=1)
        temporal_v = self.temporal_module(spatial_features) 
        
        if is_zero_shot:
            v = base_v
        else:
            v = base_v + self.gamma * temporal_v

        v_norm = F.normalize(v, p=2, dim=-1)
        
        # --- 2. TEXT ENCODER & PROMPTS ---
        target_classes = unseen_class_names if (is_zero_shot and unseen_class_names is not None) else self.class_names
        K = len(target_classes)

        clean_classes = [
            re.sub(r'([a-z])([A-Z])', r'\1 \2', c).lower().replace("something", "an object") 
            for c in target_classes
        ]

        text_prompts = [self.manual_prompt_template.format(c) for c in clean_classes]
        
        inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True, max_length=64, truncation=True).to(device)
        input_ids = inputs["input_ids"]

        if "attention_mask" in inputs:
            attn_mask = inputs["attention_mask"]
        else:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0
            attn_mask = (input_ids != pad_token_id).long()

        if self.prompt_type == "cocoop" and self.meta_net is not None:
            # (CoCoOp logic remains as you have it...)
            pass
        else:  
            text_outputs = self.model.get_text_features(input_ids=input_ids, attention_mask=attn_mask)

            if not isinstance(text_outputs, torch.Tensor):
                if hasattr(text_outputs, "text_embeds") and text_outputs.text_embeds is not None:
                    text_outputs = text_outputs.text_embeds
                elif hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
                    text_outputs = text_outputs.pooler_output
                    if hasattr(self.model, "text_projection"):
                        text_outputs = self.model.text_projection(text_outputs)
                else:
                    text_outputs = text_outputs[1] 
                    if hasattr(self.model, "text_projection"):
                        text_outputs = self.model.text_projection(text_outputs)
            
            if not isinstance(text_outputs, torch.Tensor):
                text_outputs = text_outputs[0]
                
            t_features = F.normalize(text_outputs, p=2, dim=-1).unsqueeze(0).expand(B, -1, -1)

        # --- 3. OUTPUTS ---
        logit_scale = self.model.logit_scale.exp() 
        logit_bias = self.model.logit_bias

        # NEW: Allow returning raw features for contrastive training
        if return_features:
            return v_norm, t_features, logit_scale, logit_bias

        logits = (torch.bmm(v_norm.unsqueeze(1), t_features.transpose(1, 2)).squeeze(1) * logit_scale) + logit_bias
        return logits