import flet as ft
import pandas as pd
from io_utils import load_dataset, load_questions
from analyzer import run_full_analysis, answer_questions
from viz import plot_missingness, plot_correlations, plot_timeseries

def main(page: ft.Page):
    page.title = "AI Data Analyst Agent (Desktop)"
    page.window_width = 1080
    page.window_height = 800

    # State
    data_bytes = None
    data_name = None
    q_bytes = None
    q_name = None
    df = None
    questions = []

    status = ft.Text("")
    results = ft.Text(selectable=True)
    charts = ft.Column(scroll=ft.ScrollMode.AUTO)
    use_llm_chk = ft.Checkbox(label="Use LLM fallback (requires OPENAI_API_KEY)", value=False)

    data_picker = ft.FilePicker(on_result=lambda e: on_data_selected(e))
    q_picker = ft.FilePicker(on_result=lambda e: on_q_selected(e))
    page.overlay.append(data_picker)
    page.overlay.append(q_picker)

    data_label = ft.Text("No data file selected.")
    q_label = ft.Text("No questions file selected.")

    def on_data_selected(e: ft.FilePickerResultEvent):
        nonlocal data_bytes, data_name
        if e.files:
            f = e.files[0]
            data_name = f.name
            with open(f.path, "rb") as fh:
                data_bytes = fh.read()
            data_label.value = f"Data: {data_name}"
        else:
            data_label.value = "No data file selected."
        page.update()

    def on_q_selected(e: ft.FilePickerResultEvent):
        nonlocal q_bytes, q_name
        if e.files:
            f = e.files[0]
            q_name = f.name
            with open(f.path, "rb") as fh:
                q_bytes = fh.read()
            q_label.value = f"Questions: {q_name}"
        else:
            q_label.value = "No questions file selected."
        page.update()

    def run_analysis(e):
        nonlocal df, questions
        charts.controls.clear()
        results.value = ""
        status.value = "Loading files..."
        page.update()

        if not data_bytes or not data_name:
            status.value = "Please select a data file first."
            page.update()
            return

        # Load dataset
        df, msg = load_dataset(data_name, data_bytes)
        if df is None:
            status.value = f"Failed to load dataset: {msg}"
            page.update()
            return

        status.value = f"{msg} | Rows: {len(df)} Cols: {df.shape[1]} | Analyzing..."
        page.update()

        # Load questions (optional but recommended)
        questions = []
        if q_bytes and q_name:
            questions = load_questions(q_name, q_bytes)

        # Clean column names quickly
        df.columns = [str(c).strip() for c in df.columns]

        # Run analysis
        try:
            report = run_full_analysis(df)
        except Exception as ex:
            report = {"error": f"analysis failed: {ex}"}

        # Build textual summary
        summary_lines = []
        if "error" in report:
            summary_lines.append(f"ERROR: {report['error']}")
        else:
            shape = report["profile"]["shape"]
            summary_lines.append(f"Dataset shape: {shape[0]} rows x {shape[1]} cols")
            # Missingness
            miss = report["profile"]["missing_counts"]
            missing_cols = {k:v for k,v in miss.items() if v>0}
            if missing_cols:
                top5 = dict(sorted(missing_cols.items(), key=lambda kv: kv[1], reverse=True)[:5])
                summary_lines.append(f"Columns with missing values (top 5): {top5}")
            # Correlations
            strongest = report["correlations"].get("strongest_pair")
            if strongest:
                summary_lines.append(
                    f"Strongest correlation: {strongest['col1']} vs {strongest['col2']} = {strongest['corr']:.3f}"
                )
            # Outliers
            outc = report["outliers"].get("outlier_count")
            if outc is not None:
                summary_lines.append(f"Outlier count (IsolationForest): {outc}")
            # PCA
            evr = report["pca"].get("explained_variance_ratio")
            if evr:
                summary_lines.append(f"PCA explained variance (first 3): {[round(x,3) for x in evr[:3]]}")
            # Clustering
            clus = report["clustering"].get("cluster_sizes")
            if clus:
                summary_lines.append(f"KMeans clusters: {clus}")

        # Charts
        try:
            png = plot_missingness(df)
            charts.controls.append(ft.Image(src_base64=png, width=600, height=350, fit=ft.ImageFit.CONTAIN))
        except Exception:
            pass

        try:
            png = plot_correlations(df)
            charts.controls.append(ft.Image(src_base64=png, width=600, height=350, fit=ft.ImageFit.CONTAIN))
        except Exception:
            pass

        try:
            png = plot_timeseries(df)
            charts.controls.append(ft.Image(src_base64=png, width=600, height=350, fit=ft.ImageFit.CONTAIN))
        except Exception:
            pass

        # Q&A
        qa = {}
        if questions:
            try:
                qa = answer_questions(df, questions, use_llm_chk.value)
            except Exception as ex:
                qa = {"__error__": f"question answering failed: {ex}"}

        # Display
        if questions and qa:
            summary_lines.append("\nAnswers to uploaded questions:")
            for q in questions:
                a = qa.get(q, "(no answer)")
                summary_lines.append(f"- Q: {q}\n  A: {a}")

        results.value = "\n".join(summary_lines) if summary_lines else "Analysis complete."
        status.value = "Done."
        page.update()

    page.add(
        ft.Row([
            ft.Column([
                ft.Text("📊 AI Data Analyst Agent", size=24, weight=ft.FontWeight.BOLD),
                ft.Row([
                    ft.ElevatedButton("Select Data (.csv/.xlsx/.pdf)", on_click=lambda _: data_picker.pick_files(allow_multiple=False)),
                    data_label
                ]),
                ft.Row([
                    ft.ElevatedButton("Select Questions (.csv/.pdf/.txt)", on_click=lambda _: q_picker.pick_files(allow_multiple=False)),
                    q_label
                ]),
                use_llm_chk,
                ft.ElevatedButton("Run Analysis", on_click=run_analysis),
                status,
                ft.Divider(),
                ft.Text("Results", size=18, weight=ft.FontWeight.BOLD),
                results,
            ], expand=1),
            ft.VerticalDivider(),
            ft.Column([
                ft.Text("Charts", size=18, weight=ft.FontWeight.BOLD),
                charts
            ], width=640, expand=False)
        ], expand=True)
    )

if __name__ == "__main__":
    ft.app(target=main)
