#  Software Defect Prediction Pipeline & Web Application

A comprehensive, production-ready platform that integrates traditional Machine Learning and Deep Learning architectures to automatically detect software defects in source code. Equipped with an intuitive Streamlit web interface, this repository serves as a practical codebase for developer onboarding and engineering interns.

---

##  Core System Components

The repository is organized into a modular 3-layer architecture, applying Software Engineering and Object-Oriented Programming:

1. **AI Core & Structural Analytics (`src/`):**
   * **Traditional ML & Learning-to-Rank (LTR):** Implements robust ranking algorithms (LambdaMART, GBDT, RankNet) optimized with the **SMOTE-PENN** resampling technique to handle highly imbalanced defect datasets and rank the riskiest source files (Top-K Defection).
   * **Deep Learning (Tree-LSTM + Attention):** Parses Java source code into Abstract Syntax Trees (AST) using `javalang`. It leverages a **Child-Sum Tree-LSTM** network coupled with a two-layer **Attention Pooling** mechanism to learn deep structural semantics directly from code hierarchies.

2. **Web Application (`app/`):**
   * Powered by **Streamlit**, providing an interactive web dashboard where users can drag-and-drop or upload `.java` files or the whole source code to get real-time defect risk scoring.

3. **Execution Pipelines (`src/pipelines/`):**
   * Isolated, CLI-driven scripts designed for cross-project and within-project defect validation.

---

## 📁 Repository Layout

```text
software-defect-predictor/
├── .venv/                 # Python virtual environment
├── app/                   # App configurations and resources
├── data/                  # Input datasets and target label CSVs
├── src/                   # Main source code directory[cite: 4]
│   ├── Artifacts/         # Directory containing trained model weights and metadata[cite: 4]
│   │   ├── clf_within_poi.joblib       # Classifier model (Random Forest/XGBoost)[cite: 4]
│   │   ├── ranker_cross_topk.joblib    # Ranker model (LightGBM)[cite: 4]
│   │   ├── scaler_within_poi.joblib    # Feature scaler (StandardScaler)[cite: 4]
│   │   ├── treelstm_within_poi.pth     # Trained Tree-LSTM PyTorch weights[cite: 4]
│   │   └── vocab_within_poi.json       # Token-to-ID vocabulary mapping[cite: 4]
│   ├── lstm/              # Tree-LSTM Deep Learning module[cite: 4]
│   │   ├── ast_parser.py  # Parses Java source files into Abstract Syntax Tree (AST) structures[cite: 4]
│   │   ├── config.py      # Global hyperparameter config, seed controls, and device (CUDA/CPU) setup[cite: 4]
│   │   ├── data_loader.py # Discovers projects and loads CSV/AST datasets[cite: 4]
│   │   ├── dataset.py     # Defines the Vocab class and the PyTorch TreeDataset[cite: 4]
│   │   ├── main.py        # Orchestrates the within-project and cross-project training pipelines[cite: 4]
│   │   ├── model.py       # Child-sum Tree-LSTM and Attention Pooling architecture[cite: 4]
│   │   └── trainer.py     # Handles pretraining, feature extraction, and ML classifier training[cite: 4]
│   ├── top_k/             # Learning-to-Rank (LTR) ranking module[cite: 4]
│   │   ├── train_ranker.py# Trains LTR ranker models (LambdaMART, GBDT, etc.) via LightGBM[cite: 4]
│   │   └── utils.py       # Implements SMOTE-PENN data balancing and extracts 20 static metrics[cite: 4]
│   └── demo.py            # Streamlit web application for interactive defect prediction and Top-K ranking[cite: 4]
├── README.md              # Project setup, execution instructions, and documentation
└── requirements.txt       # List of Python dependencies and package versions