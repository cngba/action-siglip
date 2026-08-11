import json
import os

# Thay đổi đường dẫn này trỏ tới thư mục chứa file của bạn
annotation_dir = "/root/data/labels" 

labels_json_path = os.path.join(annotation_dir, "labels.json")
test_csv_path = os.path.join(annotation_dir, "test-answers.csv")
output_txt_path = os.path.join(annotation_dir, "test_with_labels.txt")

# 1. Đọc labels.json để tạo từ điển ánh xạ (Tên Class -> ID)
with open(labels_json_path, 'r', encoding='utf-8') as f:
    name_to_id = json.load(f)

# 2. Đọc test-answers.csv và chuyển đổi sang chuẩn của validation.txt
count = 0
with open(test_csv_path, 'r', encoding='utf-8') as f_in, open(output_txt_path, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        if not line: 
            continue
        
        # Tách bằng dấu chấm phẩy
        parts = line.split(';')
        if len(parts) == 2:
            video_id = parts[0]
            template_name = parts[1]
            
            # Tra cứu ID từ từ điển
            if template_name in name_to_id:
                label_id = name_to_id[template_name]
                # Ghi ra chuẩn mới: Thêm .webm và dùng ID số
                f_out.write(f"{video_id}.webm {label_id}\n")
                count += 1

print(f"Đã chuyển đổi thành công {count} video ra file: {output_txt_path}")