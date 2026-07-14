import os
import pathlib
import joblib
import numpy as np
import pandas as pd
from scipy.stats import foldnorm, wilcoxon
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from top_k.utils import smote_penn_balance

FEATURE_COLS = [
    'wmc','dit','noc','cbo','rfc','lcom',
    'ca','ce','npm','lcom3','loc','dam',
    'moa','mfa','cam','ic','cbm','amc','max_cc','avg_cc'
]

ALGORITHMS = {
    "RankNet":       {"objective": "rank_xendcg"},
    "LambdaMART":    {"objective": "lambdarank"},
    "GBDT Regression": {"objective": "regression"},
    "Random Forests": {
        "objective": "regression",
        "boosting_type": "rf",
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
    },
}

BASE_PARAMS = {
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    "verbose": -1,
}

def load_and_preprocess_data():
    all_data = []
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename.endswith(".csv"):
                file_path = os.path.join(dirname, filename)
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip().str.lower()
                df.rename(columns={'avg(cc)': 'avg_cc', 'class': 'name', 'Class': 'name'}, inplace=True)
                df['file_name'] = df['name'] if 'name' in df.columns else 'unknown'
                df.drop(columns=['name'], inplace=True, errors='ignore')
                release_name = filename.replace('.csv', '').lower()
                df['release'] = release_name
                df['project'] = release_name.split('-')[0]
                if 'max_cc' not in df.columns:
                    df['max_cc'] = pd.NA
                all_data.append(df)
    
    full_data = pd.concat(all_data, ignore_index=True)
    
    for col in FEATURE_COLS + ['bug']:
        full_data[col] = pd.to_numeric(full_data[col], errors='coerce')
    for col in FEATURE_COLS:
        full_data[col] = full_data[col].fillna(full_data[col].median())
        
    full_data['bug'] = full_data['bug'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    x_all = full_data['bug'].values
    c, loc_param, scale = foldnorm.fit(x_all, floc=0)
    mu_hat = c * scale
    sigma_hat = scale
    
    def relevance_label(x, mu, sigma):
        if x == 0:              return 0
        elif x <= mu + sigma:   return 1
        elif x <= mu + 2*sigma: return 2
        else:                   return 3
        
    full_data['relevance'] = full_data['bug'].apply(lambda x: relevance_label(x, mu_hat, sigma_hat))
    return full_data

def make_groups(n, max_size=5000):
    groups, rem = [], n
    while rem > 0:
        g = min(max_size, rem)
        groups.append(g); rem -= g
    return groups

def train_and_score(X_train, y_train, X_test, y_test):
    train_grp = make_groups(len(y_train))
    test_grp  = make_groups(len(y_test))
    train_ds = lgb.Dataset(X_train, label=y_train, group=train_grp)
    test_ds  = lgb.Dataset(X_test,  label=y_test,  group=test_grp, reference=train_ds)
 
    scores = {}
    for name, extra_params in ALGORITHMS.items():
        params = {**BASE_PARAMS, **extra_params}
        model  = lgb.train(params, train_ds, valid_sets=[test_ds], num_boost_round=100)
        y_pred = model.predict(X_test)
        scores[name] = {
            "NDCG@3":  ndcg_score([y_test], [y_pred], k=3),
            "NDCG@5":  ndcg_score([y_test], [y_pred], k=5),
            "NDCG@10": ndcg_score([y_test], [y_pred], k=10),
            "y_pred":  y_pred,
            "model_object": model,
        }
    return scores

if __name__ == "__main__":
    full_data = load_and_preprocess_data()
    all_releases = sorted(full_data['release'].unique())
    results = {name: {"NDCG@3": [], "NDCG@5": [], "NDCG@10": []} for name in ALGORITHMS}
    last_trained_models = {}
    example_top10 = None

    for idx, test_release in enumerate(all_releases):
        test_project = test_release.split('-')[0]
        train_mask = full_data['project'] != test_project
        test_mask  = full_data['release'] == test_release
        df_train = full_data[train_mask].reset_index(drop=True)
        df_test  = full_data[test_mask].reset_index(drop=True)
     
        if len(df_test) == 0 or df_test['relevance'].nunique() < 2:
            for name in ALGORITHMS:
                for k in ["NDCG@3","NDCG@5","NDCG@10"]:
                    results[name][k].append(np.nan)
            continue
     
        X_tr_raw = df_train[FEATURE_COLS].values
        y_tr_raw = df_train['relevance'].values
        names_tr = df_train['file_name'].tolist()
        X_te = df_test[FEATURE_COLS].values
        y_te = df_test['relevance'].values
     
        X_tr, y_tr, names_tr_bal = smote_penn_balance(X_tr_raw, y_tr_raw, names_tr, k=5, beta=0.5, n_smote=200)
     
        fold_scores = train_and_score(X_tr, y_tr, X_te, y_te)
        for name, sc in fold_scores.items():
            last_trained_models[name] = sc["model_object"]
            results[name]["NDCG@3"].append(sc["NDCG@3"])
            results[name]["NDCG@5"].append(sc["NDCG@5"])
            results[name]["NDCG@10"].append(sc["NDCG@10"])
     
        if idx == len(all_releases) - 1:
            best_pred = fold_scores["LambdaMART"]["y_pred"]
            df_top10 = df_test.copy()
            df_top10['predicted_score'] = best_pred
            df_top10 = df_top10.sort_values('predicted_score', ascending=False)
            example_top10 = df_top10[['file_name','predicted_score','relevance']].head(10)

    summary = {}
    for name in ALGORITHMS:
        row = {}
        for k in ["NDCG@3","NDCG@5","NDCG@10"]:
            vals = [v for v in results[name][k] if not np.isnan(v)]
            row[k] = np.median(vals)
        row["Median_overall"] = np.median([row["NDCG@3"], row["NDCG@5"], row["NDCG@10"]])
        summary[name] = row
    
    best_model = max(summary, key=lambda n: summary[n]["Median_overall"])
    print(f"\n Mô hình tốt nhất: {best_model}")

    # Lưu model xuất sắc nhất
    output_dir = pathlib.Path("Artifacts")  
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_model in last_trained_models:
        real_model_object = last_trained_models[best_model]
        joblib.dump(real_model_object, output_dir / "ranker_cross_topk.joblib")
        print("✅ Lưu model thành công vào Artifacts/ranker_cross_topk.joblib")