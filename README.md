# AI Data Analyst Agent – Desktop

A desktop “AI Data Analyst” that accepts CSV / Excel / PDF, performs advanced analysis (correlations, PCA, outliers, clustering, basic anomaly detection, charts), ingests a questions file (CSV with `question` column or PDF), and returns precise answers. Optional LLM fallback (OpenAI) enhances free-form Q&A.

## ✨ Features
- Load data: `.csv`, `.xlsx`, `.pdf` (tables in PDF)
- Load questions: `.csv` (column `question`) or `.pdf` / `.txt`
- Advanced analysis:
  - Missingness report
  - Correlation matrix & strongest pair
  - IsolationForest outlier detection
  - PCA (variance + component loadings)
  - KMeans clustering
  - Basic time-series plot (auto-detects date/time column)
- Charts (matplotlib) embedded in the UI
- Deterministic rule-based Q&A + optional LLM (OpenAI)

## 🧰 Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
