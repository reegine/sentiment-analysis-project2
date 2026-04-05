# Indonesian Sentiment Analysis (Classic ML + IndoBERT)

This project trains and runs sentiment prediction for **Indonesian
Instagram comments** using three sentiment labels:

-   `-1` → Negative\
-   `0` → Neutral\
-   `1` → Positive

It supports two modeling approaches:

1.  **Classic Machine Learning**
    -   TF-IDF + LinearSVC
    -   Output artifact: `sentiment_pipeline.pkl`
2.  **IndoBERT Transformer Model**
    -   Model: `indobenchmark/indobert-base-p1`
    -   Output artifact folder: `output/indobert_model/`

------------------------------------------------------------------------

# 📌 Table of Contents

-   Requirements
-   First-Time Setup (Including Git LFS)
-   Data Format for Training
-   Train Model
-   Run Prediction
-   Compare Prediction vs Manual Labels
-   Quick Start Guide
-   Troubleshooting
-   File Structure
-   Notes
-   License

------------------------------------------------------------------------

# 🔧 Requirements

-   Python 3.10+ (3.11 / 3.12 recommended)
-   Git LFS (required for large model & data files)
-   Internet connection (for first IndoBERT download if training from
    scratch)

------------------------------------------------------------------------

# 🚀 First-Time Setup (Including Git LFS)

## 1️⃣ Install Git LFS

### Windows

Download from: https://git-lfs.com/ Or:

    winget install GitLFS.GitLFS

### macOS

    brew install git-lfs

### Linux

    sudo apt-get install git-lfs

------------------------------------------------------------------------

## 2️⃣ Clone Repository

    git clone https://github.com/reegine/sentiment-analysis-project2.git
    cd sentiment-analysis-project2

------------------------------------------------------------------------

## 3️⃣ Initialize Git LFS & Download Large Files

    git lfs install
    git lfs pull

------------------------------------------------------------------------

## 4️⃣ Setup Python Environment

### Windows (PowerShell)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r "Test Model 1/requirements.txt"

### macOS / Linux

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r "Test Model 1/requirements.txt"

------------------------------------------------------------------------

# 📂 Data Format for Training

Place your labeled dataset in:

    Test Model 1/data/combined_data.csv

Required Columns:

-   topic / TOPIK
-   username_posting
-   link_post
-   username_komentar
-   isi_komentar
-   Sentimen (-1, 0, 1 or negative/neutral/positive)

------------------------------------------------------------------------

# 🧠 Train Model

### IndoBERT (Default)

    cd "Test Model 1"
    python train_model.py --model-type indobert

### Classic Model

    cd "Test Model 1"
    python train_model.py --model-type classic

------------------------------------------------------------------------

# 🔮 Run Prediction

### Predict from CSV

    cd "Test Model 1"
    python predict.py --model-type auto --input data/combined_data.csv --out-prefix hasil_sentimen

### Predict from Nested JSON

    cd "Test Model 1"
    python predict.py --model-type auto --input data/raw_comments_regine.json --out-prefix hasil_sentimen_scrape

Outputs are saved in:

    Test Model 1/output/

------------------------------------------------------------------------

# 📊 Compare Prediction vs Manual Labels

    cd "Test Model 1"
    python evaluate_against_manual.py --manual data/combined_data.csv --pred output/hasil_sentimen.csv

------------------------------------------------------------------------

# ⚡ Quick Start

## Option A: Use Pre-Trained Model (Recommended)

    git clone https://github.com/reegine/sentiment-analysis-project2.git
    cd sentiment-analysis-project2
    git lfs install
    git lfs pull

    python -m venv .venv
    source .venv/bin/activate
    pip install -r "Test Model 1/requirements.txt"

    cd "Test Model 1"
    python predict.py --model-type auto --input data/combined_data.csv --out-prefix results

## Option B: Train Your Own Model

    git clone https://github.com/reegine/sentiment-analysis-project2.git
    cd sentiment-analysis-project2

    python -m venv .venv
    source .venv/bin/activate
    pip install -r "Test Model 1/requirements.txt"

    cd "Test Model 1"
    python train_model.py --model-type indobert
    python predict.py --model-type auto --input data/your_data.csv --out-prefix results

------------------------------------------------------------------------

# 🛠 Troubleshooting

### Git LFS Issues

    git lfs ls-files
    git lfs pull --include="*"

### Recreate Virtual Environment

    rm -rf .venv
    python -m venv .venv
    source .venv/bin/activate
    pip install -r "Test Model 1/requirements.txt"

------------------------------------------------------------------------

# 📁 File Structure

    sentiment-analysis-project2/
    ├── .gitattributes          # Git LFS tracking rules
    ├── .gitignore              # Ignored files
    ├── Test Model 1/
    │   ├── data/               # CSV/XLSX data files
    │   ├── output/             
    │   │   ├── indobert_model/ # Pre-trained model (downloaded via LFS)
    │   │   │   ├── model.safetensors
    │   │   │   ├── config.json
    │   │   │   └── tokenizer.json
    │   │   └── *.csv           # Prediction outputs
    │   ├── *.py                # Python scripts
    │   └── requirements.txt
    └── README.md

------------------------------------------------------------------------

# 📝 Notes

-   Git LFS is required for downloading the pre-trained model (\~500MB).
-   First download may take several minutes.
-   All predictions are saved in `Test Model 1/output/`.

------------------------------------------------------------------------

# 📄 License

Specify your license here (MIT, Apache 2.0, etc.)
