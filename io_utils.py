import io
import os
from typing import List, Tuple, Optional
import pandas as pd
import pdfplumber

def load_tabular_from_pdf(pdf_bytes: bytes) -> List[pd.DataFrame]:
    """Try to extract tables from a PDF into a list of DataFrames."""
    tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # Extract tables
            extracted = page.extract_tables()
            for tbl in extracted:
                if tbl and len(tbl) >= 2:
                    # Assume first row as header if all strings and distinct-ish
                    header = tbl[0]
                    body = tbl[1:]
                    try:
                        df = pd.DataFrame(body, columns=header)
                    except Exception:
                        # Fallback: no header
                        df = pd.DataFrame(tbl)
                    # Drop completely empty columns
                    df = df.dropna(axis=1, how="all")
                    if len(df.columns) > 0 and len(df) > 0:
                        tables.append(df)
    return tables

def load_text_from_pdf(pdf_bytes: bytes) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                text_chunks.append(text)
    return "\n".join(text_chunks)

def load_dataset(file_name: str, file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Return (DataFrame or None, message).
    For PDF: tries tables; if none, returns None with a message.
    """
    ext = os.path.splitext(file_name.lower())[1]
    try:
        if ext in [".csv"]:
            df = pd.read_csv(io.BytesIO(file_bytes))
            return df, "Loaded CSV."
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(io.BytesIO(file_bytes))
            return df, "Loaded Excel."
        elif ext == ".pdf":
            tables = load_tabular_from_pdf(file_bytes)
            if tables:
                # If multiple tables, try to concatenate if shapes compatible; else return first
                try:
                    df = pd.concat(tables, ignore_index=True)
                except Exception:
                    df = tables[0]
                return df, f"Loaded {len(tables)} table(s) from PDF."
            else:
                return None, "No tables detected in PDF. Provide a tabular file or a PDF with tables."
        else:
            return None, f"Unsupported file type: {ext}"
    except Exception as e:
        return None, f"Failed to load dataset: {e}"

def load_questions(file_name: str, file_bytes: bytes) -> List[str]:
    """
    questions.csv with a 'question' column OR a PDF with text (one question per line/paragraph).
    """
    ext = os.path.splitext(file_name.lower())[1]
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
            col = None
            for cand in ["question", "Question", "questions", "Questions"]:
                if cand in df.columns:
                    col = cand
                    break
            if col is None:
                # fallback: first column
                col = df.columns[0]
            return [str(x) for x in df[col].dropna().tolist()]
        elif ext in [".pdf"]:
            text = load_text_from_pdf(file_bytes)
            # Split by line breaks, keep non-empty
            qs = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return qs
        else:
            # treat as plain text
            txt = file_bytes.decode("utf-8", errors="ignore")
            qs = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return qs
    except Exception:
        # last resort
        try:
            txt = file_bytes.decode("utf-8", errors="ignore")
            qs = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return qs
        except Exception:
            return []
