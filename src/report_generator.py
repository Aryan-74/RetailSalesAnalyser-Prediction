"""Report generator module producing CSV and PDF executive reports via ReportLab."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

import config
from src.insights import generate_business_insights

def generate_csv_report(df: pd.DataFrame, output_path: Path) -> Path:
    """Export dataframe as a formatted CSV report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path

def generate_pdf_report(
    output_path: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None
) -> Path:
    """
    Generate an executive PDF report detailing retail sales analytics,
    ML model benchmark leaderboards, business insights, and forecasts.
    """
    if output_path is None:
        output_path = config.REPORTS_DIR / 'retail_sales_report.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = pd.read_parquet(config.CLEANED_DATA_PATH)

    # Load model metadata and metrics
    metadata = {}
    if config.METADATA_PATH.exists():
        with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    metrics_records = []
    if config.MODEL_METRICS_PATH.exists():
        with open(config.MODEL_METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_records = json.load(f)

    # Document Setup
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E3A8A'),
        fontName='Helvetica-Bold',
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=14
    )
    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E3A8A'),
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1F2937')
    )
    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # Title & Header
    story.append(Paragraph("RETAIL SALES ANALYSIS & FORECASTING REPORT", title_style))
    story.append(Paragraph(
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Kaggle Store Sales Dataset | 54 Stores, 33 Product Families",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceAfter=12))

    # Section 1: Executive KPI Summary Table
    story.append(Paragraph("1. Executive Performance Metrics", h2_style))
    total_sales = float(df['sales'].sum())
    avg_daily = float(df.groupby('date', observed=False)['sales'].sum().mean())
    total_promos = int(df['onpromotion'].sum())

    kpi_data = [
        ["Total Revenue ($)", "Avg Daily Sales ($)", "Active Stores", "Product Families", "Promotional Events"],
        [f"${total_sales:,.0f}", f"${avg_daily:,.0f}", f"{df['store_nbr'].nunique()}", f"{df['family'].nunique()}", f"{total_promos:,}"]
    ]
    t_kpi = Table(kpi_data, colWidths=[110, 110, 80, 100, 110])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 12))

    # Section 2: Machine Learning Model Benchmark
    story.append(Paragraph("2. Machine Learning Model Evaluation Leaderboard", h2_style))
    story.append(Paragraph(
        "Models were evaluated using a chronological split (Test Period: Aug 1 - Aug 15, 2017). "
        "The winning model is selected based on minimum Validation RMSLE.",
        body_style
    ))
    story.append(Spacer(1, 6))

    if metrics_records:
        test_metrics = [m for m in metrics_records if m.get('Split') == 'Test']
        model_rows = [["Model", "MAE ($)", "RMSE ($)", "RMSLE", "WAPE", "R² Score"]]
        for m in sorted(test_metrics, key=lambda x: x.get('RMSLE', 999)):
            model_rows.append([
                m.get('Model', ''),
                f"{m.get('MAE', 0):.2f}",
                f"{m.get('RMSE', 0):.2f}",
                f"{m.get('RMSLE', 0):.4f}",
                f"{m.get('WAPE', 0):.4f}",
                f"{m.get('R2', 0):.4f}"
            ])
        t_models = Table(model_rows, colWidths=[130, 75, 75, 75, 75, 80])
        t_models.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#374151')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_models)
    story.append(Spacer(1, 12))

    # Section 3: Strategic Business Insights
    story.append(Paragraph("3. Strategic Business & Operational Insights", h2_style))
    insights = generate_business_insights(df)
    insight_rows = []
    for ins in insights[:6]:
        p_cat = Paragraph(f"<b>{ins['category']}</b>", body_style)
        p_desc = Paragraph(f"<b>{ins['title']}:</b> {ins['message']}", body_style)
        insight_rows.append([p_cat, p_desc])

    t_ins = Table(insight_rows, colWidths=[90, 420])
    t_ins.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EFF6FF')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BFDBFE')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_ins)
    story.append(Spacer(1, 12))

    # Section 4: Top 5 Categories & Top 5 Stores
    story.append(Paragraph("4. Key Category & Store Performance Leaders", h2_style))
    top_fams = df.groupby('family', observed=False)['sales'].sum().sort_values(ascending=False).head(5)
    top_strs = df.groupby('store_nbr', observed=False)['sales'].sum().sort_values(ascending=False).head(5)

    lead_data = [["Rank", "Top Product Family", "Family Sales ($)", "Top Store #", "Store Sales ($)"]]
    for i in range(5):
        f_name = top_fams.index[i]
        f_val = top_fams.iloc[i]
        s_nbr = top_strs.index[i]
        s_val = top_strs.iloc[i]
        lead_data.append([
            f"#{i+1}",
            str(f_name),
            f"${f_val:,.0f}",
            f"Store {s_nbr}",
            f"${s_val:,.0f}"
        ])

    t_lead = Table(lead_data, colWidths=[40, 140, 110, 100, 120])
    t_lead.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_lead)

    # Build PDF
    doc.build(story)
    print(f"[ReportGenerator] PDF report generated at: {output_path}")
    return output_path

if __name__ == '__main__':
    print("Testing Report Generator...")
    pdf_p = generate_pdf_report()
    print("Report exists:", pdf_p.exists(), "Size:", pdf_p.stat().st_size, "bytes")
