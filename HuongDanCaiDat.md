# 🎬 SigLIP-2 Action Recognition with PEFT (LoRA)

A robust and efficient PyTorch framework for adapting Vision-Language Models (**SigLIP-2**) to Video Action Recognition tasks. This repository supports **Fully Supervised**, **Few-Shot**, and **Base-to-Novel** learning paradigms.

## 🛠️ Prerequisites

> **Note for Windows Users:** This repository and its associated bash commands are highly optimized for **Linux** environments. If you are using Windows, we highly recommend using **WSL2** (Windows Subsystem for Linux).

* **OS:** Linux (Ubuntu 20.04/22.04 recommended)
* **Python:** 3.11
* **CUDA:** 12.x - 13.0

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/cngba/action-siglip.git
cd action-siglip

```

2. **Create and activate a Python virtual environment:**

```bash
python3 -m venv ./env
source ./env/bin/activate

```

3. **Install the required core dependencies:**

```bash
pip install decord scikit-learn opencv-python-headless transformers peft tqdm pillow hf_transfer torch torchvision huggingface_hub wandb fvcore matplotlib

```

4. **Login to Hugging Face:**
Go to [Hugging Face](https://huggingface.co/) and log in or create an account if you don't have one. To generate a token, click on your profile picture in the top right corner -> **Settings** -> **Access Tokens** -> **Create new token**. Once you have your token, run:

```bash
hf auth login

```

*When prompted, select the option to **"Paste your access token"** and paste the token you just created.*

5. **Login to Weights & Biases (wandb) for tracking:**
Go to [Weights & Biases](https://www.google.com/search?q=https://wandb.ai/), log in, and retrieve your API key from your account settings. Run the following command:

```bash
wandb login

```

*Simply paste your API key when prompted and press Enter.*

---

## 📂 Data Preparation

First, install the necessary system packages to download and extract the datasets:

```bash
apt-get update
apt-get install -y unzip unrar aria2 build-essential gcc

```

### UCF101

Run the following commands to download and extract the UCF101 videos and split files. (Ensure your config points to `/root/data/UCF-101` as appropriate).

```bash
# Create target directory
mkdir -p /root/data

# Download the videos
aria2c -c \
  --check-certificate=false \
  --max-tries=0 \
  --retry-wait=30 \
  https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

# Extract the videos
unrar x UCF101.rar /root/data

# Download the train/test splits
curl -k -L -O https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

# Extract the splits
unzip UCF101TrainTestSplits-RecognitionTask.zip -d /root/data

```

### Something-Something V2 (SSv2)

1. Visit the Qualcomm Developer Network: [Something-Something V2 Dataset](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads).
2. Click on the following links to download them to your local machine:
* `Something-something_zip1`
* `Something-something_zip2`
* `Something-Something download package labels`


3. **If using WSL:** Place the downloaded files directly into your working environment.
4. **If running on an external GPU server:** Use the `scp` command to upload the downloaded `.zip` and label files from your local machine to your server.
5. **Extract the dataset:** Once the files are ready, run the following commands to merge and extract the videos directly into `/root/data`:

```bash
# Create target data directory if it doesn't exist
mkdir -p /root/data

# Unzip the downloaded zip parts
unzip "20bn-something-something-v2-\?\?.zip"

# Concatenate the extracted parts and tar extract them directly into /root/data
cat 20bn-something-something-v2-?? | tar -xvzf - -C /root/data

```

*(Make sure to also extract your labels and place them in the correct annotation directory as defined in your YAML config).*

## ⚙️ Configuration Configuration (YAML)
This project uses a centralized YAML configuration system. All parameters regarding data paths, model hyper-parameters (LoRA rank, learning rates), and training modes are defined in the `configs/` directory.

👉 **For a detailed explanation of all parameters, please refer to the [Configuration Guide](configs/README.md).**

---

## 🚀 Usage

All experiments are driven by YAML configurations located in the `configs/` directory. Replace `<mode>` with the specific target mode defined in your config file.

*(The examples below use SSv2, but you can substitute `ssv2` with `ucf101` for UCF101 experiments).*

### 1. Training

**Fully Supervised**

```bash
python train.py -cfg configs/ssv2.yaml -m <mode>

```

**Base-to-Novel**

```bash
python train.py -cfg configs/ssv2_base2novel.yaml -m <mode>

```

**Few-Shot**

```bash
python train.py -cfg configs/ssv2_fewshot.yaml -m <mode>

```

### 2. Evaluation

> **Important Note:** For the `--weights` argument, the path should point directly to the directory defined in the `checkpoint_dir` variable of your selected mode. You must strictly select either `best_model.pt` or `last_checkpoint.pt`.

**Fully Supervised**

```bash
python test.py -cfg configs/ssv2.yaml -m <mode> --weights <path_to_checkpoint>

```

**Base-to-Novel**

* To evaluate on **Base classes**:

```bash
python test.py -cfg configs/ssv2_base2novel.yaml -m <mode> --weights <path_to_checkpoint>

```

* To evaluate on **Novel classes** (Zero-shot inference on unseen classes):

```bash
python test.py -cfg configs/ssv2.yaml -m <mode_with_zero_shot> --weights <path_to_checkpoint>

```

**Few-Shot**

```bash
python test.py -cfg configs/ssv2_fewshot.yaml -m <mode> --weights <path_to_checkpoint>

```

```

```