import random
import numpy as np
import torch


SEED       = 42
EMBED_DIM  = 128
HIDDEN_DIM = 128
DROPOUT    = 0.5
LR         = 0.001
MAX_EPOCHS = 10
PATIENCE   = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Warning: imbalanced-learn not available")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available")