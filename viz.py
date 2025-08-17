import io
import base64
import matplotlib.pyplot as plt
import pandas as pd

def figure_to_png_bytes() -> bytes:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close()
    buf.seek(0)
    return buf.read()

def plot_missingness(df: pd.DataFrame) -> bytes:
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure()
    miss.plot(kind="bar")
    plt.title("Missingness by Column")
    plt.ylabel("Fraction Missing")
    return figure_to_png_bytes()

def plot_correlations(df: pd.DataFrame) -> bytes:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        plt.figure(); plt.text(0.1, 0.5, "Not enough numeric columns")
        return figure_to_png_bytes()
    corr = num.corr()
    plt.figure()
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.colorbar()
    return figure_to_png_bytes()

def plot_timeseries(df: pd.DataFrame) -> bytes:
    # Try to guess a datetime index/column
    ts_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    dtcol = None
    for c in ts_cols:
        try:
            pd.to_datetime(df[c])
            dtcol = c
            break
        except Exception:
            continue
    if dtcol is None:
        plt.figure(); plt.text(0.1, 0.5, "No obvious date/time column")
        return figure_to_png_bytes()
    df = df.copy()
    df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce")
    num = df.select_dtypes(include="number")
    plt.figure()
    if num.shape[1] == 0:
        plt.text(0.1, 0.5, "No numeric columns to plot")
    else:
        num_cols = num.columns[:1]
        for c in num_cols:
            plt.plot(df[dtcol], df[c], label=c)
        plt.legend()
        plt.title(f"Time Series ({num_cols[0]})")
        plt.xlabel(dtcol)
        plt.ylabel(num_cols[0])
    return figure_to_png_bytes()
