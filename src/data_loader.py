import os
import re
from pathlib import Path
import pandas as pd
from ast_parser import java_file_to_ast


def discover_projects(data_root: str) -> dict:
    root = Path(data_root)
    projects = {}

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        labels_dir = subdir / "labels"
        src_dir    = subdir / "source_code"
        if not labels_dir.exists():
            inner = subdir / subdir.name
            if inner.is_dir():
                labels_dir = inner / "labels"
                src_dir    = inner / "source_code"
        if not labels_dir.exists() or not src_dir.exists():
            continue

        proj = subdir.name.replace("_project_data", "")
        projects[proj] = {}

        for csv_file in sorted(labels_dir.glob("*.csv")):
            m = re.search(r"[\-_](\d[\w\.]*)\.csv$", csv_file.name)
            version = m.group(1) if m else csv_file.stem

            src_dirs = [d for d in src_dir.iterdir() if d.is_dir()]
            src_path = next((str(d) for d in src_dirs if version in d.name), None)
            if src_path is None and src_dirs:
                src_path = str(sorted(src_dirs)[0])

            if src_path:
                projects[proj][version] = {
                    "csv": str(csv_file),
                    "src": src_path
                }

    print(f"Discovered {len(projects)} projects:")
    for p, vs in projects.items():
        print(f"  {p}: {sorted(vs.keys())}")
    return projects

def _build_file_map(src_dir: str) -> dict:
    fmap = {}
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".java"):
                fmap.setdefault(f, []).append(os.path.join(root, f))
    return fmap

def _find_bug_col(df: pd.DataFrame) -> str | None:
    candidates = ("bug", "bugs", "bug_count", "defects", "label", "is_bug")
    for c in df.columns:
        if c.lower() in candidates:
            return c
    for c in reversed(df.columns):
        try:
            pd.to_numeric(df[c])
            return c
        except Exception:
            pass
    return None

def load_version(csv_path: str, src_dir: str, label="") -> list:
    df = pd.read_csv(csv_path)
    bug_col  = _find_bug_col(df)
    name_col = next((c for c in df.columns if c.lower() in ("name", "class", "filename")), df.columns[0])
    if bug_col is None:
        print(f"  WARN: no bug column in {csv_path}")
        return []

    df["is_buggy"] = (pd.to_numeric(df[bug_col], errors="coerce").fillna(0) > 0).astype(int)
    fmap    = _build_file_map(src_dir)
    records = []

    for _, row in df.iterrows():
        class_name = str(row[name_col])
        java_file  = class_name.split(".")[-1] + ".java"
        expected   = class_name.replace(".", "/") + ".java"

        path = None
        if java_file in fmap:
            for p in fmap[java_file]:
                if expected in p.replace("\\", "/"):
                    path = p; break
            if path is None:
                path = fmap[java_file][0]

        if path:
            ast = java_file_to_ast(path)
            if ast:
                records.append({"class_name": class_name,
                                "ast": ast.to_dict(),
                                "bug_label": int(row["is_buggy"])})

    n_bug = sum(r["bug_label"] for r in records)
    pct   = n_bug / max(len(records), 1) * 100
    print(f"  {label}: {len(records)} files | {n_bug} buggy ({pct:.1f}%)")
    return records

def load_all(config: dict) -> dict:
    all_data = {}
    for proj, versions in config.items():
        all_data[proj] = {}
        for ver, cfg in versions.items():
            recs = load_version(cfg["csv"], cfg["src"], f"{proj}-{ver}")
            if recs:
                all_data[proj][ver] = recs
    return all_data