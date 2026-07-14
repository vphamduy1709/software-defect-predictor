import os
import json
import joblib
import pathlib
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Đã thay đổi: Import từ các thư mục con (package)
from lstm.ast_parser import java_file_to_ast
from lstm.dataset import tree_to_tensors
from lstm.model import TreeLSTM
from top_k.utils import extract_tabular_metrics

# Cấu hình môi trường XGBoost
os.environ["XGBOOST_THREAD_SETTING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib/:" + os.environ.get("DYLD_LIBRARY_PATH", "")

# =====================================================================
# GIAO DIỆN STREAMLIT
# =====================================================================
st.set_page_config(page_title="AI Defect Predictor", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: 700; color: #1E3A8A; text-align: center; margin-bottom: 0px; }
    .sub-title { font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title"> AI Software Defect Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Phân tích đa mã nguồn Java và Xếp hạng Top-K nguy cơ lỗi</p>', unsafe_allow_html=True)

# Lấy đường dẫn gốc của project (nơi chứa demo.py)
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "Artifacts"
TAG = "within_poi" 

@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    with open(MODEL_DIR / f"vocab_{TAG}.json", "r") as f: token2id = json.load(f)
    model = TreeLSTM(vocab_size=len(token2id), embed_dim=128, hidden_dim=128, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_DIR / f"treelstm_{TAG}.pth", map_location=torch.device("cpu")))
    model.eval()
    
    scaler = joblib.load(MODEL_DIR / f"scaler_{TAG}.joblib")
    clf = joblib.load(MODEL_DIR / f"clf_{TAG}.joblib")
    ranker = joblib.load(MODEL_DIR / "ranker_cross_topk.joblib")

    return token2id, model, scaler, clf, ranker

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103623.png", width=80)
    st.markdown("### ⚙️ Trạng thái Hệ thống")
    with st.spinner("Đang nạp mô hình AI..."):
        try:
            token2id, model, scaler, clf, ranker = load_all_artifacts()
            st.success("✅ Toàn bộ Model sẵn sàng!")
        except Exception as e:
            st.error("❌ Lỗi nạp model: Vui lòng đảm bảo các file trong thư mục Artifacts đầy đủ.")
            st.stop()
    
    st.markdown("---")
    top_k = st.slider("Số lượng file Top-K cần hiển thị:", min_value=1, max_value=10, value=3)

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("#### 1. Tải lên mã nguồn")
    uploaded_files = st.file_uploader("Kéo thả các file .java vào đây", type=["java"], accept_multiple_files=True)
    
    if uploaded_files:
        results = []
        valid_files_code = {} 
        
        with st.spinner(f"🧠 AI đang quét {len(uploaded_files)} file..."):
            for file in uploaded_files:
                code_str = file.read().decode("utf-8", errors="ignore")
                valid_files_code[file.name] = code_str
                
                temp_name = f"temp_{file.name}"
                with open(temp_name, "w", encoding="utf-8") as f: f.write(code_str)
                ast_tree = java_file_to_ast(temp_name)
                if os.path.exists(temp_name): os.remove(temp_name)
                
                if ast_tree is not None:
                    nodes, edges = tree_to_tensors(ast_tree.to_dict(), token2id)
                    with torch.no_grad(): 
                        feat_128d, _ = model(nodes, edges)
                    
                    feat_20d = extract_tabular_metrics(code_str)
                        
                    results.append({
                        "file_name": file.name,
                        "feature_vector_128d": feat_128d.numpy().reshape(1, -1),
                        "feature_vector_20d": feat_20d, 
                        "ast_nodes_count": len(nodes)
                    })
                    
        if results:
            st.markdown(f"#### 2. Kết quả Xếp hạng (Top {min(top_k, len(results))})")
            
            for res in results:
                X_scaled_128d = scaler.transform(res["feature_vector_128d"])
                res["prob"] = float(clf.predict_proba(X_scaled_128d)[0, 1]) * 100
                
                try:
                    res["rank_score"] = float(ranker.predict(res["feature_vector_20d"])[0])
                except Exception:
                    res["rank_score"] = res["prob"]
            
            results_sorted = sorted(results, key=lambda x: x["rank_score"], reverse=True)
            
            for i in range(min(top_k, len(results_sorted))):
                res = results_sorted[i]
                
                with st.expander(f"🚨 Top {i+1}: {res['file_name']}", expanded=(i==0)):
                    st.metric(label="Nguy cơ tiềm ẩn Defect", value=f"{res['prob']:.2f}%", delta=f"Rank Score: {res['rank_score']:.4f}", delta_color="inverse")
                    st.progress(int(res['prob']))
                    
                    if res['prob'] >= 50: 
                        st.error("🔴 **CẢNH BÁO:** Rủi ro rất cao. Cần Code Review ngay lập tức!")
                    else: 
                        st.success("🟢 **AN TOÀN:** Cấu trúc tương đối ổn định.")
        else:
            st.error("❌ Không thể trích xuất cây AST từ các file đã tải lên.")

with col2:
    if uploaded_files and len(uploaded_files) > 0 and 'results_sorted' in locals():
        st.markdown("#### 3. Nội dung File Code (Theo thứ tự xếp hạng)")
        display_count = min(top_k, len(results_sorted))
        tabs = st.tabs([f"Top {i+1}: {results_sorted[i]['file_name']}" for i in range(display_count)])
        
        for i, tab in enumerate(tabs):
            with tab:
                st.code(valid_files_code[results_sorted[i]["file_name"]], language="java")
    else:
        st.info("👈 Hãy tải lên các file .java ở cột bên trái để bắt đầu.")