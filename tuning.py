import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
import urllib.request
import ssl
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# Import the newly modified model
from model_tuning import Siglip2FullLoRATemporalBridge
from dataset import UCF11VideoDataset




def train_and_evaluate(config, train_loader, test_loader, class_names, device):
    """
    Khởi tạo mô hình mới, train 3 epochs, và đánh giá.
    Quản lý khắt khe bộ nhớ VRAM để chống OOM.
    """
    print(f"\n{'='*50}\nBắt đầu test cấu hình: r={config['lora_r']}, modules={config['target_modules']}\n{'='*50}")
    
    # 1. Khởi tạo mô hình "sạch"
    model = Siglip2FullLoRATemporalBridge(
        model_name="google/siglip2-base-patch16-224",
        class_names=class_names,
        lora_r=config["lora_r"],
        target_modules=config["target_modules"]
    )
    model.to(device)
    
    # Chỉ tối ưu hóa các tham số yêu cầu gradient (LoRA và Temporal Adapter)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 3 # Số epochs cho mỗi vòng lặp tuning
    
    start_time = time.time()
    
    # 2. Vòng lặp Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (pixel_values, labels) in enumerate(pbar):
            # pixel_values shape: (B, T, C, H, W)
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(pixel_values) # (B, Num_Classes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
    # 3. Vòng lặp Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pixel_values, labels in tqdm(test_loader, desc="Evaluating"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            logits = model(pixel_values)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    val_acc = (correct / total) * 100 if total > 0 else 0
    elapsed_time = time.time() - start_time
    
    print(f"Validation Accuracy: {val_acc:.2f}% | Thời gian chạy: {elapsed_time:.2f}s")
    
    # 4. CHỐNG OOM: Xóa mô hình và optimizer, dọn dẹp bộ nhớ khắt khe
    del model
    del optimizer
    del criterion
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    
    return val_acc, avg_train_loss, elapsed_time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Chuẩn bị Dữ liệu
    data_dir = os.getenv("UCF11_DATA_DIR")
    if not data_dir:
        raise ValueError("Chưa thiết lập biến môi trường UCF11_DATA_DIR trong file .env")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    
    # Giảm batch size để tránh OOM khi tuning
    batch_size = 4
    
    train_dataset = UCF11VideoDataset(data_dir, processor, num_frames=8, train=True)
    test_dataset = UCF11VideoDataset(data_dir, processor, num_frames=8, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = train_dataset.classes
    print(f"Danh sách lớp ({len(class_names)}): {class_names}")
    
    # 2. Định nghĩa các siêu tham số cần test
    lora_rs = [2, 4, 8, 16]
    target_modules_list = [
        ["q_proj", "v_proj"],
        ["q_proj", "v_proj", "k_proj"],
        ["q_proj", "v_proj", "k_proj", "out_proj"]
    ]
    
    results = []
    
    # 3. Vòng lặp Tuning
    for r in lora_rs:
        for modules in target_modules_list:
            config = {
                "lora_r": r,
                "target_modules": modules,
                "num_modules": len(modules)
            }
            val_acc, train_loss, elapsed = train_and_evaluate(config, train_loader, test_loader, class_names, device)
            
            results.append({
                "lora_r": r,
                "target_modules": str(modules),
                "num_modules": len(modules),
                "val_acc": val_acc,
                "train_loss": train_loss,
                "time_sec": elapsed
            })
            
    # 4. Xuất Báo Cáo Tường Minh & Trực Quan
    df = pd.DataFrame(results)
    df.to_csv("ucf11_lora_tuning_results.csv", index=False)
    print("\nĐã lưu kết quả thô vào ucf11_lora_tuning_results.csv")
    
    # Vẽ Heatmap
    plt.figure(figsize=(10, 6))
    pivot_df = df.pivot(index='lora_r', columns='num_modules', values='val_acc')
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Validation Accuracy (%)'})
    plt.title("Tương quan giữa LoRA Rank, Số ma trận (Target Modules) và Validation Accuracy")
    plt.xlabel("Số lượng ma trận mục tiêu")
    plt.ylabel("LoRA Rank (r)")
    plt.tight_layout()
    plt.savefig("tuning_heatmap.png", dpi=300)
    print("Đã lưu biểu đồ vào tuning_heatmap.png")
    
    # In Leaderboard
    print("\n" + "*"*40)
    print("LEADERBOARD (Sắp xếp theo Validation Acc)")
    print("*"*40)
    leaderboard = df.sort_values(by="val_acc", ascending=False).reset_index(drop=True)
    for idx, row in leaderboard.iterrows():
        print(f"Top {idx+1}: Acc = {row['val_acc']:.2f}% | r = {row['lora_r']:2} | modules = {row['target_modules']} | loss = {row['train_loss']:.4f}")
    
if __name__ == "__main__":
    main()
