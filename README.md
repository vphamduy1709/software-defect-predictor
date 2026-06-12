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
defect-predictor/
├── data/                  # Sample datasets and benchmarks
├── app/
│   └── app.py             # Streamlit web application interface
├── src/                   # Main source code directory
│   ├── __init__.py
│   ├── data_utils.py      # Core data loaders, AST parsing, and SMOTE-PENN utilities
│   ├── traditional/
│   │   ├── __init__.py
│   │   └── ranker.py      # Traditional LTR models and statistical evaluations
│   ├── deep_learning/
│   │   ├── __init__.py
│   │   └── tree_lstm.py   # Tree-LSTM Cell & Neural Network architecture
│   └── pipelines/
│       ├── __init__.py
│       ├── train_topk.py  # Many-to-One CPDP experiment pipeline
│       ├── train_within.py# Within-Project defect prediction script
│       └── train_cross.py # Cross-Project (Leave-One-Out) evaluation script
├── config.json            # Centralized hyperparameter configuration file
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
