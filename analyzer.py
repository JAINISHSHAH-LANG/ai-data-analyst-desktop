import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# OPTIONAL LLM (OpenAI) for natural language Q&A
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    desc = df.describe(include="all").applymap(lambda x: x if isinstance(x, (int, float)) else str(x))
    missing = df.isna().sum().to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    return {
        "shape": list(df.shape),
        "dtypes": dtypes,
        "missing_counts": missing,
        "describe": json.loads(desc.to_json())
    }

def correlations(df: pd.DataFrame) -> Dict[str, Any]:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return {"note": "Not enough numeric columns for correlation."}
    corr = num.corr()
    # strongest pair
    tril = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
    strength = tril.abs().stack().sort_values(ascending=False)
    top = None
    if len(strength) > 0:
        (c1, c2), val = strength.index[0], strength.iloc[0]
        top = {"col1": c1, "col2": c2, "corr": float(corr.loc[c1, c2])}
    return {"matrix": json.loads(corr.to_json()), "strongest_pair": top}

def outliers_and_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    num = df.select_dtypes(include="number").dropna()
    if num.shape[0] < 10 or num.shape[1] == 0:
        return {"note": "Not enough numeric data for outlier detection."}
    iso = IsolationForest(random_state=42, n_estimators=200, contamination="auto")
    preds = iso.fit_predict(num)
    out_idx = list(np.where(preds == -1)[0])
    return {"outlier_count": len(out_idx), "outlier_indices": out_idx[:50]}

def pca_summary(df: pd.DataFrame) -> Dict[str, Any]:
    num = df.select_dtypes(include="number").dropna()
    if num.shape[1] < 2 or num.shape[0] < 3:
        return {"note": "Not enough data for PCA."}
    X = StandardScaler().fit_transform(num.values)
    pca = PCA(n_components=min(5, X.shape[1]))
    comps = pca.fit_transform(X)
    exp = pca.explained_variance_ratio_.tolist()
    return {
        "explained_variance_ratio": exp,
        "top_component_corrs": {
            f"PC{i+1}": dict(zip(num.columns, pca.components_[i]))
            for i in range(len(exp))
        }
    }

def clustering_kmeans(df: pd.DataFrame, k: int = 3) -> Dict[str, Any]:
    num = df.select_dtypes(include="number").dropna()
    if num.shape[0] < k or num.shape[1] == 0:
        return {"note": "Not enough numeric data for clustering."}
    X = StandardScaler().fit_transform(num.values)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    counts = pd.Series(labels).value_counts().to_dict()
    return {"k": k, "cluster_sizes": counts}

def infer_answers_locally(df: pd.DataFrame, questions: List[str]) -> Dict[str, str]:
    """
    Simple deterministic QA rules for common analytics questions.
    If a question doesn't match rules, we attempt a heuristic summary.
    """
    answers = {}
    num = df.select_dtypes(include="number")
    for q in questions:
        q_low = q.lower()
        try:
            if "row count" in q_low or "how many rows" in q_low:
                answers[q] = f"{len(df)}"
            elif "column count" in q_low or "how many columns" in q_low:
                answers[q] = f"{df.shape[1]}"
            elif "missing" in q_low or "null" in q_low:
                miss_cols = df.isna().sum().sort_values(ascending=False)
                top = miss_cols[miss_cols > 0].head(5).to_dict()
                answers[q] = f"Columns with missing values (top 5): {top}" if top else "No missing values."
            elif "correlation" in q_low and num.shape[1] >= 2:
                corr = num.corr().abs()
                tril = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                pair = tril.stack().sort_values(ascending=False).head(1)
                if len(pair) > 0:
                    (c1, c2), val = pair.index[0], pair.iloc[0]
                    answers[q] = f"Strongest correlation: {c1} vs {c2} = {val:.3f}"
                else:
                    answers[q] = "No correlations detected."
            elif any(kw in q_low for kw in ["mean", "average"]) and num.shape[1] > 0:
                col = num.columns[0]
                answers[q] = f"Mean of {col} = {num[col].mean():.4f}"
            elif any(kw in q_low for kw in ["max", "maximum"]) and num.shape[1] > 0:
                col = num.columns[0]
                val = num[col].max()
                row = df.loc[num[col].idxmax()].to_dict()
                answers[q] = f"Max {col} = {val}; row={row}"
            elif any(kw in q_low for kw in ["min", "minimum"]) and num.shape[1] > 0:
                col = num.columns[0]
                val = num[col].min()
                row = df.loc[num[col].idxmin()].to_dict()
                answers[q] = f"Min {col} = {val}; row={row}"
            else:
                # fallback local summary
                answers[q] = "I analyzed the data; see the summary and charts. For nuanced natural language questions, enable the LLM."
        except Exception as e:
            answers[q] = f"Could not determine from rules: {e}"
    return answers

def llm_answers(df: pd.DataFrame, questions: List[str]) -> Dict[str, str]:
    if not OPENAI_API_KEY or OpenAI is None:
        return {}
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Keep a compact JSON to avoid token bloat
    preview = {
        "columns": df.columns.tolist(),
        "types": df.dtypes.astype(str).to_dict(),
        "sample_rows": json.loads(df.head(30).to_json(orient="records")),
        "describe": json.loads(df.describe(include="all").to_json())
    }
    answers = {}
    for q in questions:
        prompt = (
            "You are a data analyst. Given the dataset summary (json) and the user's question, "
            "answer precisely with numbers and column names when relevant. If unknown, say so.\n\n"
            f"DATASET_JSON:\n{json.dumps(preview)[:12000]}\n\n"
            f"QUESTION:\n{q}\n"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Be concise, numeric when possible, and reference exact columns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            answers[q] = resp.choices[0].message.content.strip()
        except Exception as e:
            answers[q] = f"LLM error: {e}"
    return answers

def run_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    res = {
        "profile": basic_profile(df),
        "correlations": correlations(df),
        "outliers": outliers_and_anomalies(df),
        "pca": pca_summary(df),
        "clustering": clustering_kmeans(df, k=3),
    }
    return res

def answer_questions(df: pd.DataFrame, questions: List[str], use_llm: bool) -> Dict[str, str]:
    local = infer_answers_locally(df, questions)
    if use_llm:
        llm = llm_answers(df, questions)
        # Merge, preferring LLM if it produced something
        for q in questions:
            if q in llm and llm[q] and not llm[q].startswith("LLM error"):
                local[q] = llm[q]
    return local
