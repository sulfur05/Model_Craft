# utils/reporting.py
from io import BytesIO
from datetime import datetime
from typing import List, Optional
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import pandas as pd

def _fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _draw_multiline_text(c: canvas.Canvas, x: float, y: float, text: str, max_width: float, leading: float = 12):
    words = text.split()
    line = ""
    lines = []
    for w in words:
        test = f"{line} {w}".strip()
        if c.stringWidth(test, "Helvetica", 10) <= max_width:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    text_obj = c.beginText(x, y)
    text_obj.setFont("Helvetica", 10)
    for ln in lines:
        text_obj.textLine(ln)
    c.drawText(text_obj)
    return y - leading * len(lines)

def _write_kv_table(c: canvas.Canvas, x: float, y: float, items: dict, col_gap: float, max_width: float, size=10):
    c.setFont("Helvetica", size)
    for k, v in items.items():
        key = f"{k}:"
        val = str(v)
        c.drawString(x, y, key)
        c.drawRightString(x + max_width - col_gap, y, val)
        y -= size + 4
        if y < 80:
            c.showPage()
            y = A4[1] - 40
            c.setFont("Helvetica", size)
    return y

def _draw_dataframe_snippet(c: canvas.Canvas, x: float, y: float, df: pd.DataFrame, max_cols: int = 6, row_limit: int = 6):
    c.setFont("Helvetica", 9)
    cols = df.columns[:max_cols].tolist()
    header = " | ".join(cols)
    c.drawString(x, y, header)
    y -= 12
    rows = df.head(row_limit).to_dict(orient="records")
    for r in rows:
        line = " | ".join(str(r.get(c, "")) for c in cols)
        c.drawString(x, y, line)
        y -= 10
        if y < 80:
            c.showPage()
            y = A4[1] - 40
            c.setFont("Helvetica", 9)
    return y

def create_pdf_report_bytes(
    *,
    bundle: dict,
    dataset_summary_text: Optional[str] = None,
    eda_figs: Optional[List] = None,
    shap_figs: Optional[List] = None,
    preprocessing_summary: Optional[str] = None,
    model_params: Optional[dict] = None,
    model_metrics: Optional[dict] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    confusion_fig: Optional[plt.Figure] = None,
    pred_vs_actual_fig: Optional[plt.Figure] = None,
    sample_predictions: Optional[pd.DataFrame] = None,
    max_plots_per_page: int = 2,
) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    # Cover
    title = bundle.get("trained_model_name", "ModelCraft Report")
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    c.drawRightString(width - margin, y, f"Model saved: {bundle.get('created_at', '')}")
    y -= 30

    # Metadata + schema
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Model & dataset overview")
    y -= 16
    meta = {
        "Model name": bundle.get("trained_model_name"),
        "Version": bundle.get("version"),
        "Task": bundle.get("task_type"),
        "Target": bundle.get("target_column"),
        "Dataset shape": bundle.get("dataset_shape"),
        "Feature count": len(bundle.get("feature_columns", [])),
    }
    y = _write_kv_table(c, margin, y, meta, col_gap=20, max_width=width - 2 * margin)

    # Feature list (short)
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Feature schema (first 40 features)")
    y -= 14
    features = bundle.get("feature_columns", [])[:40]
    if features:
        feat_text = ", ".join(map(str, features))
        y = _draw_multiline_text(c, margin, y, feat_text, max_width=width - 2 * margin)
    else:
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, "No feature list available.")
        y -= 12

    # Dataset summary
    if dataset_summary_text:
        y -= 8
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Dataset summary / EDA insights")
        y -= 14
        y = _draw_multiline_text(c, margin, y, dataset_summary_text, max_width=width - 2 * margin)

    # Insert EDA figures
    if eda_figs:
        for fig in eda_figs:
            if y < margin + 180:
                c.showPage()
                y = height - margin
            img_bytes = _fig_to_png_bytes(fig)
            img = ImageReader(BytesIO(img_bytes))
            iw, ih = img.getSize()
            max_w = width - 2 * margin
            scale = min(1.0, max_w / iw)
            draw_w = iw * scale
            draw_h = ih * scale
            c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
            y -= draw_h + 10

    # Preprocessing summary (text)
    if preprocessing_summary:
        if y < margin + 80:
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Preprocessing summary")
        y -= 14
        y = _draw_multiline_text(c, margin, y, preprocessing_summary, max_width=width - 2 * margin)

    # Model params + metrics
    if model_params or model_metrics:
        if y < margin + 120:
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Model hyperparameters")
        y -= 14
        if model_params:
            y = _draw_multiline_text(c, margin, y, str(model_params), max_width=width - 2 * margin)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Model training metrics")
        y -= 14
        if model_metrics:
            y = _draw_multiline_text(c, margin, y, str(model_metrics), max_width=width - 2 * margin)

    # Diagnostic plots (confusion, pred vs actual)
    if confusion_fig is not None:
        if y < margin + 200:
            c.showPage()
            y = height - margin
        img_bytes = _fig_to_png_bytes(confusion_fig)
        img = ImageReader(BytesIO(img_bytes))
        iw, ih = img.getSize()
        max_w = width - 2 * margin
        scale = min(1.0, max_w / iw)
        draw_w = iw * scale
        draw_h = ih * scale
        c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
        y -= draw_h + 10

    if pred_vs_actual_fig is not None:
        if y < margin + 200:
            c.showPage()
            y = height - margin
        img_bytes = _fig_to_png_bytes(pred_vs_actual_fig)
        img = ImageReader(BytesIO(img_bytes))
        iw, ih = img.getSize()
        max_w = width - 2 * margin
        scale = min(1.0, max_w / iw)
        draw_w = iw * scale
        draw_h = ih * scale
        c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
        y -= draw_h + 10

    # SHAP figs
    if shap_figs:
        for fig in shap_figs:
            if y < margin + 200:
                c.showPage()
                y = height - margin
            img_bytes = _fig_to_png_bytes(fig)
            img = ImageReader(BytesIO(img_bytes))
            iw, ih = img.getSize()
            max_w = width - 2 * margin
            scale = min(1.0, max_w / iw)
            draw_w = iw * scale
            draw_h = ih * scale
            c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
            y -= draw_h + 10

    # Comparison table snippet
    if comparison_df is not None:
        if y < margin + 120:
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Model comparison (top rows)")
        y -= 14
        y = _draw_dataframe_snippet(c, margin, y, comparison_df, max_cols=6, row_limit=6)

    # Sample predictions
    if sample_predictions is not None:
        if y < margin + 120:
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Sample predictions")
        y -= 14
        y = _draw_dataframe_snippet(c, margin, y, sample_predictions, max_cols=6, row_limit=6)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()