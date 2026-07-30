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
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(input_size=dim, hidden_size=dim // 2, bidirectional=True, batch_first=True)
        self.q_pool = nn.Parameter(torch.randn(dim, 1))
        nn.init.trunc_normal_(self.q_pool, std=0.02)

    def forward(self, x):
        x_transpose = x.transpose(1, 2)
        x_conv = F.gelu(self.conv(x_transpose)).transpose(1, 2)
        x_attn, _ = self.attn(x, x, x)
        x_fuse = x + self.alpha * x_conv + self.beta * x_attn
        x_ln = self.norm(x_fuse)
        x_gru, _ = self.gru(x_ln)
        
        C_sqrt = (x_gru.size(-1) ** 0.5)
        attn_logits = (x_gru @ self.q_pool) / C_sqrt
        a = F.softmax(attn_logits, dim=1)
        v = (x_gru * a).sum(dim=1)
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

    def forward(self, pixel_values, unseen_class_names=None, is_zero_shot=False):
        B, T, C_img, H, W = pixel_values.shape
        device = pixel_values.device
        
        # --- 1. VISION ENCODER (FIXED FOR SIGLIP MAP POOLING) ---
        pixel_values = pixel_values.view(B * T, C_img, H, W)
        
        # get_image_features handles ViT encoding, MAP pooling, AND visual_projection automatically
        image_embeds = self.model.get_image_features(pixel_values=pixel_values)

        # Safely extract the tensor if a ModelOutput object is returned
        if not isinstance(image_embeds, torch.Tensor):
            if hasattr(image_embeds, "pooler_output") and image_embeds.pooler_output is not None:
                image_embeds = image_embeds.pooler_output
            elif hasattr(image_embeds, "image_embeds"):
                image_embeds = image_embeds.image_embeds
            else:
                image_embeds = image_embeds[0] 
        
        # Reshape the fully processed spatial features back to temporal sequences
        spatial_features = image_embeds.view(B, T, -1) # type: ignore

        # 1. Bắt buộc L2 Normalize TỪNG FRAME trước để tránh các frame nhiễu (magnitude lớn) nuốt chửng vector
        spatial_features_norm = F.normalize(spatial_features, p=2, dim=-1)

        
        # Temporal Aggregation
        # 1. Base Feature từ Spatial Mean Pooling (Giữ nguyên khả năng ZSL nguyên bản)
        base_v = spatial_features_norm.mean(dim=1)
        
        # 2. Dynamic Feature từ Temporal Module
        temporal_v = self.temporal_module(spatial_features) 
        
        # 3. Blending bằng Learnable Gate (gamma bắt đầu = 0)
        v = base_v + self.gamma * temporal_v
        v_norm = F.normalize(v, p=2, dim=-1)
        
        # --- 2. TEXT ENCODER & PROMPTS ---
        # FIX: Lấy đúng danh sách target classes được truyền từ test.py
        target_classes = unseen_class_names if (is_zero_shot and unseen_class_names is not None) else self.class_names
        K = len(target_classes)

        # FIX: Tách CamelCase và chuẩn hóa chữ thường (VD: ApplyEyeMakeup -> apply eye makeup)
        clean_classes = [re.sub(r'([a-z])([A-Z])', r'\1 \2', c).lower() for c in target_classes]
        text_prompts = [self.manual_prompt_template.format(c) for c in clean_classes]

        # Ép buộc max_length=64 để Positional Embedding của SigLIP không bị lệch pha do padding động
        inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True, max_length=64, truncation=True).to(device)
        input_ids = inputs["input_ids"]

        # Safely extract attention_mask or create it manually if missing
        if "attention_mask" in inputs:
            attn_mask = inputs["attention_mask"]
        else:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0  # Fallback if tokenizer has no pad_token_id set
            attn_mask = (input_ids != pad_token_id).long()
        # attn_mask = inputs["attention_mask"]

        if self.prompt_type == "cocoop" and self.meta_net is not None:
            delta_v = self.meta_net(v) # (B, M, D)
            M = self.meta_net.num_prompt_tokens        
                
            word_embeds = self.model.text_model.embeddings.token_embedding(input_ids) # (K, L, D)
            L = word_embeds.size(1)
            
            word_embeds = word_embeds.unsqueeze(0).expand(B, -1, -1, -1)
            delta_v_expand = delta_v.unsqueeze(1).expand(-1, K, -1, -1)
            
            combined_embeds = torch.cat([delta_v_expand, word_embeds], dim=2).view(B * K, M + L, -1)
            
            seq_length = M + L
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
            position_embeds = self.model.text_model.embeddings.position_embedding(position_ids)
            dynamic_prompts = combined_embeds + position_embeds
            
            base_mask = attn_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * K, L)
            prompt_mask = torch.ones((B * K, M), dtype=base_mask.dtype, device=device)
            full_mask = torch.cat([prompt_mask, base_mask], dim=1)
            
            # text_outputs = self.model.text_model(inputs_embeds=dynamic_prompts, attention_mask=full_mask)
            
            # # Extract robust pooled representation
            # if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            #     pooled_text = text_outputs.pooler_output
            # else:
            #     seq_lengths = full_mask.sum(dim=-1).to(torch.long) - 1
            #     batch_indices = torch.arange(text_outputs.last_hidden_state.shape[0], device=device)
            #     pooled_text = text_outputs.last_hidden_state[batch_indices, seq_lengths, :]

            # --- CUSTOM FORWARD PASS VỚI CHUNKING ĐỂ CHỐNG OOM ---
        
            # 1. Biến đổi attention mask từ 2D sang 4D
            extended_attention_mask = full_mask[:, None, None, :].to(dtype=dynamic_prompts.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dynamic_prompts.dtype).min
            
            # 2. Chia nhỏ batch khổng lồ (B*K) thành các khối nhỏ để chạy
            chunk_size = 128  # Nếu vẫn bị OOM, hãy hạ số này xuống 64 hoặc 32
            all_pooled_text = []
            
            for i in range(0, dynamic_prompts.size(0), chunk_size):
                chunk_prompts = dynamic_prompts[i : i + chunk_size]
                chunk_mask = extended_attention_mask[i : i + chunk_size]
                
                chunk_outputs = self.model.text_model.encoder(
                    inputs_embeds=chunk_prompts,
                    attention_mask=chunk_mask
                )
                chunk_last_hidden_state = self.model.text_model.final_layer_norm(chunk_outputs[0])
                
                # Trích xuất text feature cho chunk
                chunk_seq_lengths = full_mask[i : i + chunk_size].sum(dim=-1).to(torch.long) - 1
                chunk_batch_indices = torch.arange(chunk_last_hidden_state.shape[0], device=device)
                chunk_pooled = chunk_last_hidden_state[chunk_batch_indices, chunk_seq_lengths, :]
                
                all_pooled_text.append(chunk_pooled)
                
            # 3. Gộp các khối lại thành tensor hoàn chỉnh
            pooled_text = torch.cat(all_pooled_text, dim=0)
                
            # Crucial: Apply SigLIP's text projection to align with the visual space
            if hasattr(self.model, "text_projection"):
                raw_text_feats = self.model.text_projection(pooled_text)
            else:
                raw_text_feats = pooled_text
                
            t_features = F.normalize(raw_text_feats, p=2, dim=-1).view(B, K, -1)

        else:  # Manual prompt flow
            text_outputs = self.model.get_text_features(input_ids=input_ids, attention_mask=attn_mask)

            # Safely extract the pooled and projected text embeddings
            if not isinstance(text_outputs, torch.Tensor):
                if hasattr(text_outputs, "text_embeds") and text_outputs.text_embeds is not None:
                    text_outputs = text_outputs.text_embeds
                elif hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
                    # If we only have pooler_output, we must manually apply the SigLIP text projection
                    text_outputs = text_outputs.pooler_output
                    if hasattr(self.model, "text_projection"):
                        text_outputs = self.model.text_projection(text_outputs)
                else:
                    # Fallback for tuples: [0] is usually last_hidden_state, [1] is pooler_output
                    text_outputs = text_outputs[1] 
                    if hasattr(self.model, "text_projection"):
                        text_outputs = self.model.text_projection(text_outputs)
            
            if not isinstance(text_outputs, torch.Tensor):
                text_outputs = text_outputs[0]
                
            t_features = F.normalize(text_outputs, p=2, dim=-1).unsqueeze(0).expand(B, -1, -1)

        # --- 3. LOGITS ---
        logit_scale = self.model.logit_scale.exp() 
        logit_bias = self.model.logit_bias
        logits = (torch.bmm(v_norm.unsqueeze(1), t_features.transpose(1, 2)).squeeze(1) * logit_scale) + logit_bias
        
        # return F.softmax(logits, dim=-1) if is_zero_shot else logits
        return logits