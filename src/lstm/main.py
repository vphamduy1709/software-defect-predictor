import pandas as pd
import torch
import lstm.ast_parser as ast_parser
from lstm.dataset import tree_to_tensors, Vocab
from lstm.data_loader import discover_projects, load_all
from lstm.trainer import run_experiment
from lstm.ast_parser import java_file_to_ast
from lstm.config import DEVICE


def run_within_project(all_data: dict, output_dir="model_within"):
    print("\n" + "=" * 60)
    print("WITHIN-PROJECT (train older → test latest)")
    print("=" * 60)

    all_metrics = []
    for proj, versions in all_data.items():
        sorted_v = sorted(versions.keys())
        if len(sorted_v) < 2:
            print(f"\n  [{proj}] < 2 versions, skipped")
            continue

        train_v = sorted_v[:-1]
        test_v  = sorted_v[-1]
        train_data = [r for v in train_v for r in versions[v]]
        test_data  = versions[test_v]

        n_bug_tr = sum(r["bug_label"] for r in train_data)
        n_bug_te = sum(r["bug_label"] for r in test_data)
        print(f"\n[{proj}]  Train: {train_v} → {len(train_data)} files ({n_bug_tr} buggy)")
        print(f"          Test:  {test_v}   → {len(test_data)} files ({n_bug_te} buggy)")

        if len(test_data) < 10 or n_bug_te == 0:
            print("  Test set quá nhỏ hoặc không có bug, skip")
            continue

        _, _, _, _, metrics = run_experiment(
            train_data, test_data, output_dir, tag=f"within_{proj}"
        )
        all_metrics.append({"project": proj, **metrics})

    if all_metrics:
        df = pd.DataFrame(all_metrics).set_index("project")
        print("\n" + "─" * 60)
        print("WITHIN-PROJECT — TỔNG KẾT")
        print("─" * 60)
        print(df.round(4).to_string())
        print("\nTrung bình:")
        print(df.mean().round(4).to_string())

    return all_metrics

def run_cross_project(all_data: dict, output_dir="model_cross"):
    print("\n" + "=" * 60)
    print("CROSS-PROJECT — Leave-One-Project-Out")
    print("=" * 60)

    projects = list(all_data.keys())
    if len(projects) < 2:
        print("Cần ≥ 2 projects, skip")
        return []

    all_metrics = []
    for target in projects:
        test_v    = sorted(all_data[target].keys())[-1]
        test_data = all_data[target][test_v]

        train_data = [r for proj, versions in all_data.items()
                      if proj != target for records in versions.values() for r in records]

        n_bug_tr = sum(r["bug_label"] for r in train_data)
        n_bug_te = sum(r["bug_label"] for r in test_data)
        print(f"\n[Target: {target}]  Train: {len(train_data)} ({n_bug_tr} buggy) | Test({test_v}): {len(test_data)} ({n_bug_te} buggy)")

        if len(train_data) < 50 or len(test_data) < 10 or n_bug_te == 0:
            print("  Không đủ dữ liệu, skip")
            continue

        _, _, _, _, metrics = run_experiment(
            train_data, test_data, output_dir, tag=f"cross_{target}"
        )
        all_metrics.append({"target": target, **metrics})

    if all_metrics:
        df = pd.DataFrame(all_metrics).set_index("target")
        print("\n" + "─" * 60)
        print("CROSS-PROJECT — TỔNG KẾT")
        print("─" * 60)
        print(df.round(4).to_string())
        print("\nTrung bình:")
        print(df.mean().round(4).to_string())

    return all_metrics

# =============================================================
# 13. PREDICT SINGLE FILE
# =============================================================
def predict_file(file_path: str, model, vocab: Vocab, scaler, clf) -> tuple[int | None, float | None]:
    ast = java_file_to_ast(file_path)
    if ast is None:
        return None, None
    nodes, edges = tree_to_tensors(ast.to_dict(), vocab)
    model.eval()
    with torch.no_grad():
        feat, _ = model(nodes.to(DEVICE), edges.to(DEVICE))
    X = scaler.transform(feat.squeeze(0).cpu().numpy().reshape(1, -1))
    prob = clf.predict_proba(X)[0, 1]
    return (1 if prob >= 0.5 else 0), float(prob)

#
if __name__ == "__main__":
    BASE_DIR = "/kaggle/input/datasets/minhquang333/sdp-data"

    print("=" * 60)
    print("DISCOVERING DATA")
    print("=" * 60)
    config   = discover_projects(BASE_DIR)
    all_data = load_all(config)
    
    print(f"\nTotal parse errors: {ast_parser.parse_errors}")

    total_files = sum(len(r) for vs in all_data.values() for r in vs.values())
    print(f"Total files loaded: {total_files}")

    # within_results = run_within_project(all_data, output_dir="model_within")
    cross_results = run_cross_project(all_data, output_dir="model_cross")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)