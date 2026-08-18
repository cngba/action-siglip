# 📝 Configuration Guide (YAML)

The `configs/` directory serves as the control center for this project. We use a centralized YAML configuration system to manage parameters across 3 main learning paradigms: **Fully Supervised**, **Few-Shot**, and **Base-to-Novel** (Zero-Shot Generalization) on 2 datasets (UCF101 and SSv2).

There are 6 primary configuration files:
* `ucf101.yaml` / `ssv2.yaml`: For Fully Supervised learning.
* `ucf101_fewshot.yaml` / `ssv2_fewshot.yaml`: For Few-Shot learning (2, 4, 8, 16 shots).
* `ucf101_base2novel.yaml` / `ssv2_base2novel.yaml`: For Base-to-Novel / Zero-Shot learning.

Each YAML file is divided into two core sections: `data` (Global) and `modes` (Experiment-specific).

---

## 1. 📂 Data Block: Global Data Definitions
The `data` section defines how the DataLoader locates, reads, and splits the videos. Depending on the `setting`, these variables dictate the data pipeline.

### Common Variables:
* `dataset`: The target dataset (`ucf101` or `ssv2`). This determines which internal processing pipeline is used.
* `base_dir`: The root directory containing all raw video files (`.avi` or `.webm`).
* `annotation_dir`: The directory containing standard split files (e.g., `.json` for SSv2).
* `num_segments`: Number of frames sampled per video (typically `8`).

### Mode-Specific Variables:

**A. Fully Supervised Mode (`ucf101.yaml` / `ssv2.yaml`)**
* `split`: (UCF101 only) Selects split 1, 2, or 3[cite: 5].
* `test_file` & `test_splits_dir`: (SSv2 only). Since the official SSv2 test set lacks labels, these variables point the model to a custom `.txt` file containing ground-truth labels for accurate evaluation[cite: 5].

**B. Few-Shot Mode (`*_fewshot.yaml`)**
* `setting: "few_shot"`: Flags the DataLoader to read from custom `.txt` files rather than the default logic[cite: 5].
* `splits_dir`: Directory containing the few-shot text splits[cite: 5].
* `train_few_shot`: The specific training file (e.g., `train1_few_shot_16.txt`)[cite: 5].
* `val_few_shot`: The file used for validation and Early Stopping (e.g., `val1.txt` for UCF101, or `validation.txt` for SSv2)[cite: 5].

**C. Base-to-Novel Mode (`*_base2novel.yaml`)**
* `setting: "base2novel"`: Activates the class-splitting pipeline[cite: 5].
* `train_base` / `val_base`: Training and validation files for Base (seen) classes[cite: 5].
* `val_novel`: Test set file for Novel (unseen) classes used in Zero-Shot inference[cite: 5].
* `zero_shot_splits`: **CRITICAL**. Contains two arrays: `seen_class_names` and `unseen_class_names`[cite: 5]. SigLIP relies heavily on these arrays to compile exact Text Prompts into accurate Text Embeddings for Zero-Shot alignment.

---

## 2. ⚙️ Modes Block: Training Strategies & Hyperparameters

Inside the `modes` block, you can define multiple experimental setups. You can trigger a specific mode using the `-m` flag in your command line.

### Core Variables:
* `checkpoint_dir` & `log_file`: Where to save the `.pt` weights and training logs[cite: 5].
* `prompt_type` & `manual_prompt_template`: The sentence template for the Text Encoder (e.g., `"A video of a person performing {}"`)[cite: 5].

### Breakdown of the 4 Typical Modes:

#### 1. Baseline No LoRA (Frozen Base)
Trains only the Custom Temporal Head while keeping the entire SigLIP backbone frozen.
* `unfreeze_backbone: false`[cite: 5]
* `lora_r: 0`: Disables LoRA[cite: 5].
* `lr_base: 0.0`: Prevents learning in the original parameters[cite: 5].
* `lr_head: 1.0e-3`: Standard learning rate for the newly initialized Head[cite: 5].

#### 2. Baseline With LoRA (PEFT)
Keeps the backbone frozen but injects trainable LoRA matrices into the Attention layers.
* `lora_r: 16`: The rank of the LoRA matrices[cite: 5].
* `lora_alpha: 16.0`: Scaling factor (usually set equal to `r`)[cite: 5].
* `lora_target_modules: ["q_proj", "v_proj"]`: Injects LoRA into the Query and Value projections of the Vision/Text Encoders[cite: 5].
* `lr_base: 5.0e-5` (or `1.0e-4`): A smaller learning rate for LoRA weights[cite: 5].
* `lr_head: 1.0e-3`: A larger learning rate for the Temporal Head (**Multi-rate optimization**)[cite: 5].

#### 3. Baseline FFT (Full Fine-Tuning)
Unfreezes 100% of the parameters (~380M parameters). This is a highly sensitive mode prone to Out-Of-Memory (OOM) errors and Catastrophic Forgetting.
* `unfreeze_backbone: true`: **Unlocks the Backbone.**[cite: 5]
* `batch_size: 8` / `accumulation_steps: 4`: Reduces the physical batch size to prevent OOM while using Gradient Accumulation to maintain an Effective Batch Size of 32[cite: 5].
* `lr_base: 1.0e-6`: **Must be kept extremely small** to avoid destroying pre-trained SigLIP representations[cite: 5].
* `warmup_epochs: 5`: Linear Warmup is **MANDATORY** here to prevent gradient overshoot during the initial epochs[cite: 5].

#### 4. Zero-Shot Evaluation Modes (e.g., `zero_shot_baseline_with_lora`)
Modes containing the string `"zero_shot"` in their name are specifically designed for evaluating the model on **Novel (unseen) classes** during testing.
* `checkpoint_dir`: Must point to the exact same directory as the corresponding training mode (e.g., pointing back to `./exp/.../baseline_with_lora`) so the script can locate the correct saved weights and store the test logs/confusion matrices in the right place[cite: 5].
* **Architecture Integrity**: Model parameters (`lora_r`, `cocoop_hidden_dim`, `unfreeze_backbone`) must strictly match the training mode to ensure the state dictionary loads successfully without missing or unexpected key errors[cite: 5].
* `batch_size`: Can usually be maximized (e.g., set to 32) since gradients are not computed or stored during evaluation[cite: 5].