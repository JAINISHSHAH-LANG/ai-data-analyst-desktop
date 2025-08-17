import os
import uuid
import base64
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from io_utils import load_dataset, load_questions
from analyzer import run_full_analysis, answer_questions
from viz import plot_missingness, plot_correlations, plot_timeseries

app = FastAPI(title="AI Data Analyst Agent API 🚀")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload(file: UploadFile) -> str:
    """Save uploaded file locally and return path."""
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path


@app.post("/analyze/")
async def analyze_data(
    data_file: UploadFile = File(...),
    q_file: UploadFile = File(None),
    use_llm: bool = Form(False),
):
    """
    Upload data + optional questions.
    Returns analysis + answers + charts (base64).
    """
    # Save and load dataset
    data_path = save_upload(data_file)
    df, msg = load_dataset(data_file.filename, open(data_path, "rb").read())
    if df is None:
        return JSONResponse(status_code=400, content={"error": f"Failed to load dataset: {msg}"})

    # Load questions if provided
    questions = []
    if q_file:
        q_path = save_upload(q_file)
        questions = load_questions(q_file.filename, open(q_path, "rb").read())

    # Run full analysis
    try:
        report = run_full_analysis(df)
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {ex}"})

    # Generate charts
    charts = {}
    try:
        charts["missingness"] = base64.b64encode(plot_missingness(df)).decode()
    except Exception:
        charts["missingness"] = None
    try:
        charts["correlations"] = base64.b64encode(plot_correlations(df)).decode()
    except Exception:
        charts["correlations"] = None
    try:
        charts["timeseries"] = base64.b64encode(plot_timeseries(df)).decode()
    except Exception:
        charts["timeseries"] = None

    # Answer questions
    qa = {}
    if questions:
        qa = answer_questions(df, questions, use_llm)

    return {
        "message": msg,
        "profile": report["profile"],
        "correlations": report["correlations"],
        "outliers": report["outliers"],
        "pca": report["pca"],
        "clustering": report["clustering"],
        "charts": charts,
        "questions": questions,
        "answers": qa,
    }
