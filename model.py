import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from transformers import SiglipModel, AutoProcessor

class LoRALinear(nn.Module):
    """
    Tích hợp ma trận LoRA hạng thấp song song với lớp tuyến tính đã đóng băng[cite: 1, 2].
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

        # Khởi tạo ma trận A theo phân phối Kaiming Uniform, B bằng 0
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_linear(x)
        lora_output = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_output + lora_output * self.scaling


class HybridTemporalModule(nn.Module):
    """
    Mô đun Không-Thời gian Lai chuyển giao từ học không gian sang học không-thời gian.
    """
    def __init__(self, dim=768, num_heads=8, kernel_size=3):
        super().__init__()
        
        # 1. Nhánh trích xuất đặc trưng cục bộ (Conv1D)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # 2. Nhánh trích xuất tương quan toàn cục (MHSA)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Các hệ số vô hướng học được khởi tạo bằng 0.1
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        
        self.norm = nn.LayerNorm(dim)
        
        # Mạng GRU hai chiều với kích thước ẩn C/2 để đầu ra đồng nhất C
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            bidirectional=True,
            batch_first=True
        )
        
        # Vector truy vấn cho Attention Pooling
        self.q_pool = nn.Parameter(torch.randn(dim, 1))
        nn.init.trunc_normal_(self.q_pool, std=0.02)

    def forward(self, x):
        # X shape: (B, T, C)
        
        # Nhánh cục bộ: X_conv = GELU(Conv1D(X))
        x_transpose = x.transpose(1, 2) # (B, C, T)
        x_conv = F.gelu(self.conv(x_transpose)).transpose(1, 2) # (B, T, C)
        
        # Nhánh toàn cục: X_attn = MHSA(X)
        x_attn, _ = self.attn(x, x, x)
        
        # Nhánh kết nối tắt và Tích hợp có trọng số động
        x_fuse = x + self.alpha * x_conv + self.beta * x_attn
        
        # Chuẩn hóa LayerNorm
        x_ln = self.norm(x_fuse)
        
        # Làm mịn thông tin bằng BiGRU
        x_gru, _ = self.gru(x_ln) # Đầu ra sẽ có chiều (B, T, dim//2 * 2) = (B, T, C)
        
        # Attention Pooling
        # a = softmax((X_gru * q_pool) / sqrt(C))
        C_sqrt = (x_gru.size(-1) ** 0.5)
        attn_logits = (x_gru @ self.q_pool) / C_sqrt # (B, T, 1)
        a = F.softmax(attn_logits, dim=1) # (B, T, 1)
        
        # Nén chuỗi thời gian thành vector duy nhất v
        v = (x_gru * a).sum(dim=1) # (B, C)
        
        return v


class MetaNet(nn.Module):
    """
    Mạng Meta-Net sinh các token ngữ cảnh động từ đặc trưng video.
    """
    def __init__(self, dim, num_prompt_tokens=4):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.dim = dim
        # Cấu trúc MLP gồm 2 tầng tuyến tính kết hợp hàm kích hoạt ReLU
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_prompt_tokens * dim)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v):
        # Đầu vào v: (B, C), Đầu ra: (B, M, C)
        B = v.size(0)
        delta_v = self.net(v)
        return delta_v.view(B, self.num_prompt_tokens, self.dim)


class Siglip2ActionModel(nn.Module):
    """
    Hệ thống nhận dạng hành vi người dựa trên SigLIP 2, LoRA, và Không-Thời gian Lai.
    """
    def __init__(
        self, 
        model_name="google/siglip2-base-patch16-224", 
        class_names=None, 
        lora_r=4, 
        lora_alpha=8.0,
        num_prompt_tokens=4
    ):
        super().__init__()
        if class_names is None:
            raise ValueError("class_names list required.")
        self.class_names = class_names
        
        # Tải mô hình nền tảng SigLIP 2
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if hasattr(self.model.config, "text_config") and self.model.config.text_config is not None:
            if isinstance(self.model.config.text_config, dict):
                embedding_dim = self.model.config.text_config.get("hidden_size", 768)
            else:
                embedding_dim = getattr(self.model.config.text_config, "hidden_size", 768)
        else:
            embedding_dim = getattr(self.model.config, "hidden_size", 768)

        # 1. Tích hợp LoRA cho Vision và Text Encoder (Q, K, V, O)
        self._apply_lora_to_encoder(self.model.vision_model.encoder, r=lora_r, alpha=lora_alpha, name="Vision")
        self._apply_lora_to_encoder(self.model.text_model.encoder, r=lora_r, alpha=lora_alpha, name="Text")

        # Đóng băng toàn bộ trọng số gốc, chỉ huấn luyện tham số LoRA
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # 2. Khởi tạo Mô đun Không-Thời gian Lai
        self.temporal_module = HybridTemporalModule(dim=embedding_dim)
        
        # 3. Khởi tạo Mạng sinh câu nhắc động
        self.meta_net = MetaNet(dim=embedding_dim, num_prompt_tokens=num_prompt_tokens)

    def _apply_lora_to_encoder(self, encoder, r, alpha, name):
        """Áp dụng LoRA vào các ma trận Q, K, V, O của Self-Attention."""
        num_injected = 0
        for layer in encoder.layers:
            attn = layer.self_attn
            if hasattr(attn, "q_proj"):
                attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha)
                attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha)
                attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha)
                attn.out_proj = LoRALinear(attn.out_proj, r=r, alpha=alpha)
                num_injected += 4
        print(f"[{name} Encoder] {num_injected} LoRA matrices injected.")

    def forward(self, pixel_values, unseen_class_names=None, is_zero_shot=False):
        B, T, C_img, H, W = pixel_values.shape
        device = pixel_values.device
        
        # --- BỘ MÃ HÓA HÌNH ẢNH & MÔ ĐUN THỜI GIAN ---
        pixel_values = pixel_values.view(B * T, C_img, H, W)
        
        # Trích xuất đặc trưng không gian qua Vision Encoder (đã nhúng LoRA)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        spatial_features = self.model.vision_model.head(vision_outputs.last_hidden_state) 
        spatial_features = spatial_features.view(B, T, -1) 
        
        # Đưa qua mô đun không-thời gian lai để lấy đặc trưng video toàn cục v (Công thức 3.14)
        v = self.temporal_module(spatial_features) # (B, D)
        
        # [CÔNG THỨC 3.14]: Chuẩn hóa L2 đặc trưng video v_i thành \widetilde{v}_i
        v_norm = F.normalize(v, p=2, dim=-1) # (B, D)
        
        # Xác định tập nhãn văn bản sử dụng
        target_classes = unseen_class_names if is_zero_shot and unseen_class_names is not None else self.class_names
        K = len(target_classes)
        
        # --- CƠ CHẾ SINH CÂU NHẮC ĐỘNG & BỘ MÃ HÓA VĂN BẢN ---
        # Sinh các token ngữ cảnh động từ v: (B, M, D)
        delta_v = self.meta_net(v) 
        M = self.meta_net.num_prompt_tokens
        
        # Chuẩn bị văn bản nhãn tĩnh (sử dụng target_classes mới)
        inputs = self.processor(text=target_classes, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        # Lấy embeddings tĩnh của nhãn từ SigLIP Text Model: (K, L, D)
        with torch.no_grad():
            word_embeds = self.model.text_model.embeddings.token_embedding(input_ids)
        L = word_embeds.size(1)
        
        # Ghép nối prompt động và nhãn cho từng video trong batch
        word_embeds = word_embeds.unsqueeze(0).expand(B, -1, -1, -1)
        delta_v_expand = delta_v.unsqueeze(1).expand(-1, K, -1, -1)
        
        # Chuỗi prompt tối ưu P_k(v): (B, K, M+L, D)
        dynamic_prompts = torch.cat([delta_v_expand, word_embeds], dim=2)
        dynamic_prompts = dynamic_prompts.view(B * K, M + L, -1)
        
        # Cập nhật Attention Mask
        base_mask = attn_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * K, L)
        prompt_mask = torch.ones((B * K, M), dtype=base_mask.dtype, device=device)
        full_mask = torch.cat([prompt_mask, base_mask], dim=1)
        
        # Đưa qua Text Encoder thu được vector t_{k,i} (Công thức 3.15)
        text_outputs = self.model.text_model(inputs_embeds=dynamic_prompts, attention_mask=full_mask)
        
        # [CÔNG THỨC 3.15]: Chuẩn hóa L2 vector nhãn văn bản thành \widetilde{t}_{k,i}
        t_features = F.normalize(text_outputs.pooler_output, p=2, dim=-1) # (B*K, D)
        t_features = t_features.view(B, K, -1) # (B, K, D)
        
        # --- [CÔNG THỨC 3.16]: TÍNH ĐỘ TƯƠNG ĐỒNG VÀ PHÂN PHỐI XÁC SUẤT ---
        # Lấy hằng số tỉ lệ e^\tau từ SigLIP
        logit_scale = self.model.logit_scale.exp() 
        
        # Tích vô hướng giữa \widetilde{v}_i và \widetilde{t}_{k,i} nhân với e^\tau
        # v_norm: (B, 1, D) @ t_features.T: (B, D, K) -> similarity logits: (B, K)
        logits = torch.bmm(v_norm.unsqueeze(1), t_features.transpose(1, 2)).squeeze(1) * logit_scale
        
        if is_zero_shot:
            # Trong kịch bản Zero-Shot Evaluation, tính toán trực tiếp xác suất Softmax (Công thức 3.16)
            probabilities = F.softmax(logits, dim=-1)
            return probabilities
            
        return logits