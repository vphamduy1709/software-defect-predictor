import os, copy, json, joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import SEED, LR, MAX_EPOCHS, PATIENCE, DEVICE, EMBED_DIM, HIDDEN_DIM, DROPOUT, HAS_IMBLEARN, HAS_XGB
from dataset import Vocab, TreeDataset
from model import TreeLSTM

if HAS_IMBLEARN:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
if HAS_XGB:
    from xgboost import XGBClassifier

def pretrain(model, tr_loader, val_loader, output_dir: str, tag: str):
    ckpt = f"{output_dir}/ckpt_{tag}.pt"
    opt  = optim.RMSprop(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, factor=0.5)

    use_amp  = torch.cuda.is_available()
    amp_ctx  = torch.amp.autocast("cuda") if use_amp else None
    grad_sc  = torch.amp.GradScaler("cuda") if use_amp else None

    best_val  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    start      = 0

    if os.path.exists(ckpt):
        saved = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        start      = saved["epoch"] + 1
        best_val   = saved["best_val"]
        no_improve = saved["no_improve"]
        best_state = copy.deepcopy(model.state_dict())
        print(f"  Resumed epoch {saved['epoch']}, best_val={best_val:.4f}")

    for epoch in range(start, MAX_EPOCHS):
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            nodes_b = batch["nodes"].squeeze(0).to(DEVICE)
            edges_b = batch["edges"].squeeze(0).to(DEVICE)
            opt.zero_grad()
            if use_amp:
                with amp_ctx:
                    _, loss = model(nodes_b, edges_b)
                grad_sc.scale(loss).backward()
                grad_sc.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                grad_sc.step(opt); grad_sc.update()
            else:
                _, loss = model(nodes_b, edges_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nodes_b = batch["nodes"].squeeze(0).to(DEVICE)
                edges_b = batch["edges"].squeeze(0).to(DEVICE)
                _, loss = model(nodes_b, edges_b)
                val_loss += loss.item()

        avg_tr  = tr_loss  / len(tr_loader)
        avg_val = val_loss / max(len(val_loader), 1)
        sched.step(avg_val)
        print(f"  Epoch {epoch+1:02d}/{MAX_EPOCHS} | train={avg_tr:.4f} | val={avg_val:.4f}")

        if avg_val < best_val - 1e-5:
            best_val   = avg_val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "best_val": best_val, "no_improve": no_improve}, ckpt)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("  Early stopping.")
                break

    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    X, y = [], []
    for batch in loader:
        nodes_b = batch["nodes"].squeeze(0).to(DEVICE)
        edges_b = batch["edges"].squeeze(0).to(DEVICE)
        feat, _ = model(nodes_b, edges_b)
        X.append(feat.squeeze(0).cpu().numpy())
        y.append(batch["label"].item())
    return np.array(X), np.array(y)

def build_classifier(X_tr, y_tr):
    if HAS_IMBLEARN:
        try:
            X_r, y_r = SMOTETomek(random_state=SEED).fit_resample(X_tr, y_tr)
        except Exception:
            try:
                X_r, y_r = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
            except Exception:
                X_r, y_r = X_tr, y_tr
    else:
        X_r, y_r = X_tr, y_tr

    n_pos = (y_r == 1).sum()
    n_neg = (y_r == 0).sum()
    print(f"  Resampled: {n_pos} positive / {n_neg} negative")

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1
    )
    estimators = [("rf", rf)]

    if HAS_XGB:
        pos_w = max(1.0, n_neg / max(n_pos, 1))
        xgb = XGBClassifier(
            n_estimators=300,
            scale_pos_weight=pos_w,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=SEED,
            verbosity=0,
            n_jobs=-1
        )
        estimators.append(("xgb", xgb))

    clf = VotingClassifier(estimators, voting="soft") if len(estimators) > 1 else rf
    clf.fit(X_r, y_r)
    return clf

def evaluate(y_true, y_pred, y_prob, title="") -> dict:
    print(f"\n  {'─'*36}")
    if title:
        print(f"  {title}")
    m = {
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        m["AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        m["AUC"] = float("nan")
    for k, v in m.items():
        print(f"  {k:10s}: {v:.4f}")
    return m

def run_experiment(train_data, test_data, output_dir: str, tag: str) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    vocab = Vocab(min_freq=1).build(train_data)
    labels = [x["bug_label"] for x in train_data]
    try:
        tr_idx, val_idx = train_test_split(
            range(len(train_data)), test_size=0.15,
            stratify=labels, random_state=SEED
        )
    except ValueError:
        tr_idx  = list(range(len(train_data)))
        val_idx = tr_idx[: max(1, len(tr_idx) // 10)]

    nw  = 2 if torch.cuda.is_available() else 0
    pin = torch.cuda.is_available()

    def make_loader(data, shuffle=False):
        return DataLoader(TreeDataset(data, vocab), batch_size=1,
                          shuffle=shuffle, num_workers=nw, pin_memory=pin)

    tr_loader   = make_loader([train_data[i] for i in tr_idx],  shuffle=True)
    val_loader  = make_loader([train_data[i] for i in val_idx])
    full_loader = make_loader(train_data)
    test_loader = make_loader(test_data)

    model = TreeLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM, DROPOUT).to(DEVICE)
    print(f"\n  Pretraining ({len(tr_idx)} train / {len(val_idx)} val)...")
    model = pretrain(model, tr_loader, val_loader, output_dir, tag)

    print("  Extracting features...")
    X_tr, y_tr = extract_features(model, full_loader)
    X_te, y_te = extract_features(model, test_loader)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    print("  Training classifier...")
    clf    = build_classifier(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    try:
        y_prob = clf.predict_proba(X_te)[:, 1]
    except Exception:
        y_prob = y_pred.astype(float)

    metrics = evaluate(y_te, y_pred, y_prob)

    torch.save(model.state_dict(), f"{output_dir}/treelstm_{tag}.pth")
    joblib.dump(scaler, f"{output_dir}/scaler_{tag}.joblib")
    joblib.dump(clf,    f"{output_dir}/clf_{tag}.joblib")
    with open(f"{output_dir}/vocab_{tag}.json", "w") as f:
        json.dump(vocab.token2id, f)
    with open(f"{output_dir}/model_cfg_{tag}.json", "w") as f:
        json.dump({"vocab_size": len(vocab),
                   "embed_dim": EMBED_DIM,
                   "hidden_dim": HIDDEN_DIM}, f)

    print(f"  Artifacts → '{output_dir}/'")
    return model, vocab, scaler, clf, metrics