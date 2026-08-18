import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel
from peft import get_peft_model, LoraConfig
import av

class SigLIP2HardDemoPt:
    def __init__(self, model_id_or_path, pt_weights_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[*] Loading processor and base model from: {model_id_or_path}")
        
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        base_model = AutoModel.from_pretrained(
            model_id_or_path, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        
        print("[*] Setting up LoRA configuration...")
        self.model = get_peft_model(base_model, lora_config)
        
        if pt_weights_path and os.path.exists(pt_weights_path):
            print(f"[*] Loading weights from: {pt_weights_path}")
            state_dict = torch.load(pt_weights_path, map_location=self.device)
            
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"[!] Checkpoint not found at: {pt_weights_path}")
            
        self.model.to(self.device)
        self.model.eval()

    def load_and_sample_frames(self, video_path, num_frames=8):
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        if total_frames > 0:
            indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
        else:
            indices = list(range(num_frames))

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_image())
            if len(frames) == num_frames:
                break
        return frames

    def predict(self, video_path, candidate_labels):
        frames = self.load_and_sample_frames(video_path, num_frames=8)
        
        inputs = self.processor(
            text=candidate_labels,
            images=frames,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
            image_embeds = outputs.image_embeds 
            text_embeds = outputs.text_embeds   
            
            if image_embeds.ndim > 2:
                image_embeds = image_embeds.mean(dim=0, keepdim=True)
            elif image_embeds.shape[0] > 1 and image_embeds.shape[0] == len(frames):
                image_embeds = image_embeds.mean(dim=0, keepdim=True)

            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * self.model.base_model.model.logit_scale.exp()
            probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()[0]

        results = sorted(zip(candidate_labels, probs), key=lambda x: x[1], reverse=True)
        return results


def load_candidate_labels(labels_json_path):
    """Loads all template labels from labels.json."""
    with open(labels_json_path, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    return list(labels_dict.keys())


def get_ground_truth(video_id, labels_dir):
    """
    Searches validation.json and train.json for the video_id and returns the template label.
    """
    for split in ["validation.json", "train.json"]:
        file_path = os.path.join(labels_dir, split)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if str(item.get("id")) == str(video_id):
                        # SSv2 annotations store template under 'template' key
                        return item.get("template", item.get("label", "Unknown"))
    return "Ground truth not found in train/val annotations"


if __name__ == "__main__":
    BASE_DIR = "/root/data/20bn-something-something-v2"
    LABELS_DIR = "/root/data/labels"
    
    sample_video_name = "1.webm"
    video_path = os.path.join(BASE_DIR, sample_video_name)
    video_id = os.path.splitext(sample_video_name)[0] # Extracts "1" from "1.webm"

    MODEL_ID = "google/siglip2-base-patch16-224" 
    PT_PATH = "/root/action-siglip/exp/siglip2_peft/ssv2/baseline_with_lora_16/20260815_091753_baseline_with_lora_16_8f/best_model.pt"

    labels_json_path = os.path.join(LABELS_DIR, "labels.json")
    candidate_labels = load_candidate_labels(labels_json_path)

    # Lookup ground truth
    ground_truth = get_ground_truth(video_id, LABELS_DIR)

    demo = SigLIP2HardDemoPt(MODEL_ID, PT_PATH)
    
    if os.path.exists(video_path):
        print(f"\n[*] Processing video: {video_path}")
        results = demo.predict(video_path, candidate_labels)
        
        print(f"\n========================================")
        print(f" GROUND TRUTH:  {ground_truth}")
        print(f"========================================")
        print("\n--- TOP 5 PREDICTED ACTIONS ---")
        for rank, (label, prob) in enumerate(results[:5], start=1):
            is_match = " (CORRECT)" if label.strip() == ground_truth.strip() else ""
            print(f"{rank}. {label}: {prob * 100:.2f}%{is_match}")
    else:
        print(f"[!] Video file not found at: {video_path}")