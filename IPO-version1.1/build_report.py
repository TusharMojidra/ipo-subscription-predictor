"""Build the final PDS PBL project report (IPO_Project_Report.docx) - redesigned.

Run it with:  python build_report.py

Design: branded cover + certificate (matching the reference doc), a clickable
table of contents, colored headings, styled result tables, callout boxes,
the project's real charts embedded, and a footer with page numbers.
"""

import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = "IPO_Project_Report.docx"
FIG_DIR = "reports/figures"

# Brand palette (matches the dashboard)
PURPLE = "7C3AED"
PINK = "DB2777"
TEAL = "0D9488"
DARK = "1F1B2E"
LIGHT_TINT = "F3E8FF"
PINK_TINT = "FDF2F8"
GREY_TINT = "F4F4F5"
WHITE = "FFFFFF"

TEAM = [
    ("1", "Khandhar Piyush", "240130107057"),
    ("2", "Khandla Ajay", "240130107058"),
    ("3", "Khodada Ajay", "240130107059"),
    ("4", "Meet", "240130107065"),
    ("5", "Meghanathi Milan", "240130107066"),
    ("6", "Mojidra Tushar", "240130107071"),
]

PROJECT_TITLE = "IPO SUBSCRIPTION & LISTING GAIN PREDICTION USING MACHINE LEARNING"


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def shade_paragraph(p, fill):
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def bottom_border(p, color=PURPLE, size="14"):
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), size)
    b.set(qn("w:space"), "4")
    b.set(qn("w:color"), color)
    pBdr.append(b)
    pPr.append(pBdr)


def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_cell_text(cell, text, bold=False, color=DARK, size=10.5, white_on=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string(color)
    if white_on:
        shade_cell(cell, white_on)
    return p


def style_table(tbl, header_fill=PURPLE, zebra=GREY_TINT, bold_first_col=False):
    """Purple header row + zebra stripes, all borders from 'Table Grid'."""
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, cell in enumerate(tbl.rows[0].cells):
        set_cell_text(cell, cell.text, bold=True, color=WHITE, size=10.5, white_on=header_fill)
    for ri, row in enumerate(tbl.rows[1:], start=1):
        for ci, cell in enumerate(row.cells):
            bold = bold_first_col and ci == 0
            if ri % 2 == 0:
                shade_cell(cell, zebra)
            if bold:
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.bold = True


def add_hyperlink(paragraph, url, text):
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hl = OxmlElement("w:hyperlink")
    hl.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    c = OxmlElement("w:color"); c.set(qn("w:val"), PURPLE); rPr.append(c)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rPr.append(u)
    run.append(rPr)
    t = OxmlElement("w:t"); t.text = text; run.append(t)
    hl.append(run)
    paragraph._p.append(hl)


def _field(paragraph, instr):
    run = paragraph.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(it); run._r.append(f2)
    return run


def add_toc(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = 'TOC \\o "1-2" \\h \\z \\u'
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    r2 = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = ("(Table of contents — in Word, right-click here and choose "
              "“Update Field”, or press Ctrl+A then F9.)")
    r2.append(t)
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(it); run._r.append(f2)
    p._p.append(r2); p._p.append(f3)


def add_footer(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("IPO Predictor · PBL Report · Page ")
    r.font.size = Pt(8.5); r.font.color.rgb = RGBColor.from_string("8A8A96")
    _field(p, "PAGE")
    r = p.add_run(" of ")
    r.font.size = Pt(8.5); r.font.color.rgb = RGBColor.from_string("8A8A96")
    _field(p, "NUMPAGES")


# --------------------------------------------------------------------------- #
# Content helpers
# --------------------------------------------------------------------------- #
def h1(doc, text):
    p = doc.add_heading(text, level=1)
    for r in p.runs:
        r.font.color.rgb = RGBColor.from_string(PURPLE)
    bottom_border(p, PURPLE)
    return p


def h2(doc, text):
    p = doc.add_heading(text, level=2)
    for r in p.runs:
        r.font.color.rgb = RGBColor.from_string(TEAL)
    return p


def para(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.color.rgb = RGBColor.from_string(PURPLE)
    p.add_run(text)
    return p


def bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def callout(doc, title, body, fill=LIGHT_TINT, accent=PURPLE, title_color=PURPLE):
    """A shaded one-cell box that stands out as a 'key insight'."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    shade_cell(cell, fill)
    # colored border around the box
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "10")
        el.set(qn("w:color"), accent)
        borders.append(el)
    tcPr.append(borders)
    cell.text = ""
    p1 = cell.paragraphs[0]
    r = p1.add_run(title)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(title_color)
    r.font.size = Pt(11)
    p2 = cell.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r2 = p2.add_run(body)
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = RGBColor.from_string(DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


def add_figure(doc, filename, caption, width=5.9):
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string("6B7280")


def centered(doc, text, size, bold=True, color=DARK, space_after=6, fill=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string(color)
    if fill:
        shade_paragraph(p, fill)
    return p


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for sname, size, color in (("Heading 1", 16, PURPLE), ("Heading 2", 13, TEAL),
                               ("Heading 3", 11.5, PINK)):
        st = doc.styles[sname]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string(color)
        st.font.bold = True

    # -------------------------------------------------------------- #
    # PAGE 1 - BRANDED COVER
    # -------------------------------------------------------------- #
    for _ in range(2):
        doc.add_paragraph()
    centered(doc, "Government Engineering College Gandhinagar", 22, color=DARK)
    centered(doc, "Department of Computer Engineering", 12, bold=False, color="6B7280")
    doc.add_paragraph()
    centered(doc, "PBL (Problem Based Learning)", 16, color=WHITE, fill=PURPLE)
    centered(doc, "Under subject of", 12, bold=False, color="6B7280")
    centered(doc, "Python for Data Science", 15, color=TEAL)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(PROJECT_TITLE)
    r.bold = True
    r.font.size = Pt(17)
    r.font.color.rgb = RGBColor.from_string(DARK)
    bottom_border(p, PINK, "18")
    doc.add_paragraph()
    centered(doc, "Branch: CE-A      ·      Sem: 5      ·      Academic year: 2026-27",
             12.5, bold=False, color=DARK)

    doc.add_page_break()

    # -------------------------------------------------------------- #
    # PAGE 2 - CERTIFICATE
    # -------------------------------------------------------------- #
    centered(doc, "Government Engineering College Gandhinagar", 18, color=DARK)
    doc.add_paragraph()
    centered(doc, "CERTIFICATE", 16, color=PURPLE)
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("This is to certify that the project report entitled ")
    r = p.add_run(f'"{PROJECT_TITLE}"'); r.bold = True
    p.add_run(" Submitted for the course: Python for Data Science (PDS)")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("is a bonafide record of authentic project work carried out by the "
              "following students under group in partial fulfillment of the "
              "requirements for the curriculum:")
    doc.add_paragraph()

    tbl = doc.add_table(rows=1, cols=3)
    hdr = tbl.rows[0].cells
    for i, h in enumerate(["Sr. No.", "Student Name", "Enrollment No."]):
        hdr[i].text = h
    for row in TEAM:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    style_table(tbl)

    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Signatures:"); r.bold = True
    p.add_run("                                                                                    Date :")

    doc.add_page_break()

    # -------------------------------------------------------------- #
    # TABLE OF CONTENTS (interactive in Word)
    # -------------------------------------------------------------- #
    h1(doc, "Table of Contents")
    add_toc(doc)
    doc.add_page_break()

    # -------------------------------------------------------------- #
    # REPORT BODY
    # -------------------------------------------------------------- #
    h1(doc, "1. Project Title")
    p = doc.add_paragraph()
    r = p.add_run(f'"{PROJECT_TITLE}"'); r.bold = True; r.font.color.rgb = RGBColor.from_string(PURPLE)

    h1(doc, "2. Team Group Details")
    tbl = doc.add_table(rows=1, cols=3)
    for i, h in enumerate(["Sr. No.", "Name", "Enrollment No."]):
        tbl.rows[0].cells[i].text = h
    for row in TEAM:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    style_table(tbl)
    doc.add_paragraph()

    h1(doc, "3. Abstract")
    para(doc, "Every week in India, retail investors decide whether to apply for an "
              "Initial Public Offering (IPO) using little more than grey-market "
              "rumours. This project replaces guesswork with data. Using a Kaggle "
              "dataset of 652 Indian IPOs listed between 2010 and 2026, we build and "
              "compare machine-learning models for two complementary predictions: "
              "(1) the subscription tier an IPO will achieve (Low, Medium or High) and "
              "(2) its day-1 listing gain percentage. After cleaning the data to 648 "
              "usable records and engineering log-transformed and demand-mix features, "
              "we evaluate every model on an out-of-time split (train on IPOs up to "
              "2023, test on unseen 2024-2026 IPOs) so the results honestly reflect "
              "predicting the future.")
    callout(doc, "⭐ Key results at a glance",
            "Subscription-tier model: ~62% accuracy on never-seen IPOs.  "
            "Listing-gain model: out-of-time R² = +0.29 (up from −0.20 with raw "
            "features) with a mean absolute error of ~17.8 points.  "
            "648 IPOs · 16 years of Indian market data · deployed as an interactive dashboard.",
            fill=LIGHT_TINT, accent=PURPLE)

    h1(doc, "4. Introduction")
    para(doc, "An IPO is the first sale of a company's shares to the public. In "
              "India's retail-heavy market, IPO applications are extremely popular, "
              "yet most investors have no reliable signal about how an issue will "
              "perform on listing day. Subscription numbers (how many times the issue "
              "is oversubscribed by institutional, high-net-worth and retail "
              "investors) and listing gains (the first-day price move) are the two "
              "outcomes investors care about most.")
    para(doc, "This project treats both outcomes as machine-learning problems: "
              "subscription tier is a three-class classification problem, and listing "
              "gain is a regression problem. We train models on 16 years of historical "
              "Indian IPO records, evaluate them honestly on time-held-out data, and "
              "package the winners in a user-friendly web application that anyone can "
              "use to check an upcoming IPO.")

    h1(doc, "5. Problem Statement")
    h2(doc, "5.1 Business Problem")
    bullet(doc, "Retail investors have no reliable way to judge an IPO before applying - they depend on grey-market premium (GMP) rumours.")
    bullet(doc, "Misjudging demand leads to missed allocations in hot issues or capital stuck in weak ones.")
    bullet(doc, "A data-driven demand and gain forecast would let investors make evidence-based decisions instead of rumour-based ones.")
    h2(doc, "5.2 Data Science Problem")
    bullet(doc, "Given pre-IPO facts (issue size, offer price, year), predict the subscription tier: Low, Medium or High.")
    bullet(doc, "Given post-subscription facts (QIB, HNI, RII and total oversubscription), predict the day-1 listing gain percentage.")
    bullet(doc, "Both predictions must be evaluated on IPOs the model has never seen, i.e. a time-based (out-of-time) split.")

    h1(doc, "6. Literature Review")
    para(doc, "We studied similar public projects and papers to position our work and "
              "borrow proven techniques:")
    tbl = doc.add_table(rows=1, cols=4)
    for i, h in enumerate(["Work", "Approach", "Task", "Reported result"]):
        tbl.rows[0].cells[i].text = h
    for row in [
        ("Predicting-Listing-Gains (GitHub)", "TensorFlow / Keras classifier", "Gain vs no-gain (binary)", "69% test accuracy"),
        ("IPO-prediction-model (GitHub)", "Calibrated Random Forest, log features, nested CV", "Profit vs loss (binary)", "68.75% accuracy, AUC 0.70"),
        ("IPO-PerformanceAnalysis (GitHub)", "Logistic Regression + Gradient Boosting", "Profit vs loss (binary)", "LR 73.2%, GBM 80.4%"),
        ("IPO-Underpricing (GitHub, US data)", "RF / GBM / SVM / NN + macro features", "Underpriced vs overpriced", "RF 76%"),
        ("'Prediction of IPO Subscription' (paper)", "Logistic Regression", "IPO subscription", ">90% accuracy reported"),
        ("Sonsare et al. (2023)", "Artificial Neural Network", "IPO underperformance", "up to 68.11% accuracy"),
    ]:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    style_table(tbl)
    doc.add_paragraph()
    callout(doc, "💡 What we borrowed from the literature",
            "Almost all comparable work frames the problem as binary profit/loss "
            "classification - ours is deliberately harder (3-class tier + exact gain %). "
            "Log-transformed subscription features consistently help, and QIB "
            "(institutional) demand is the top feature everywhere - exactly what our "
            "own feature-importance analysis confirms.",
            fill=PINK_TINT, accent=PINK, title_color=PINK)

    h1(doc, "7. Dataset")
    h2(doc, "7.1 Source")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run("Kaggle - ")
    add_hyperlink(p, "https://www.kaggle.com/datasets/karanammithul/ipo-data-india-2010-2025",
                  "'IPO Data India 2010-2026'")
    p.add_run(". The raw file contains 652 Indian IPO records spanning 2010 to May 2026.")
    h2(doc, "7.2 Column Descriptions")
    tbl = doc.add_table(rows=1, cols=2)
    for i, h in enumerate(["Column", "Description"]):
        tbl.rows[0].cells[i].text = h
    for c, d in [
        ("Date", "Date when the IPO opened"),
        ("IPO_Name", "Company launching the IPO"),
        ("Issue_Size (crores)", "Total issue size in INR crores"),
        ("QIB / HNI / RII", "Investor-category subscription (times oversubscribed)"),
        ("Total", "Overall subscription ratio"),
        ("Offer Price / List Price", "Price band / listing price"),
        ("Listing Gain", "% gain/loss on day 1 (our regression target)"),
        ("CMP (BSE/NSE) / Current Gains", "Current market performance"),
    ]:
        cells = tbl.add_row().cells
        cells[0].text = c
        cells[1].text = d
    style_table(tbl, bold_first_col=True)
    doc.add_paragraph()
    h2(doc, "7.3 Cleaning Summary")
    bullet(doc, "Raw records: 652. After dropping 4 unusable rows: 648 usable IPOs.")
    bullet(doc, "Formatting symbols (commas, currency signs, %) stripped from numeric fields.")
    bullet(doc, "Missing values filled; extreme outliers winsorized so a handful of extreme IPOs do not distort training.")
    add_figure(doc, "correlation_heatmap.png",
               "Figure 1 - Correlation heatmap: QIB demand correlates most strongly with day-1 gains.")

    h1(doc, "8. Methodology")
    para(doc, "The pipeline is implemented in Python (pandas, scikit-learn, matplotlib, "
              "seaborn, shap, plotly) and runs end-to-end from raw CSV to deployed "
              "dashboard. The steps below describe what was actually built.")
    h2(doc, "8.1 Data Collection")
    para(doc, "The Kaggle CSV is loaded into a pandas DataFrame (652 rows). The raw "
              "features include issue size, investor subscription ratios (QIB, HNI, "
              "RII, Total), offer price, list price and the listing-gain target.")
    h2(doc, "8.2 Data Preprocessing & EDA")
    bullet(doc, "Cleaning: drop unusable rows, strip formatting symbols, fill missing values, winsorize outliers (648 rows remain).")
    bullet(doc, "EDA: correlation heatmaps, distribution plots, subscription-mix donut and class balance checks. Key insight: institutional (QIB) demand correlates most strongly with day-1 gains.")
    add_figure(doc, "distributions.png",
               "Figure 2 - Feature distributions before transformation (heavy right skew).")
    h2(doc, "8.3 Feature Engineering")
    bullet(doc, "log1p transformation of subscription ratios (log_QIB, log_HNI, log_RII, log_Total) and of issue size / offer price - compresses the heavy skew (recent IPOs are oversubscribed up to 326x).")
    bullet(doc, "Demand-mix ratios (e.g. QIB's share of total demand) to capture institutional vs retail composition.")
    bullet(doc, "Subscription_Class: Total subscription binned into Low (< 2x), Medium (2-10x) and High (> 10x) for classification.")
    bullet(doc, "StandardScaler normalisation so large scales (crores) do not bias linear models.")
    add_figure(doc, "class_balance.png",
               "Figure 3 - Class balance of the subscription tiers (imbalanced - handled with class_weight='balanced').")
    h2(doc, "8.4 Out-of-Time Split (honest evaluation)")
    para(doc, "Instead of a random 80/20 split, models are trained on IPOs up to 2023 "
              "(438 records) and tested on unseen 2024-2026 IPOs (210 records). This "
              "measures the only thing investors care about - predicting the future - "
              "and prevents the model from 'peeking' at recent trends.")
    h2(doc, "8.5 Subscription Tier Prediction (Classification)")
    para(doc, "Trained and compared Logistic Regression, Random Forest and Support "
              "Vector Machine, all with class_weight='balanced'. Features: log issue "
              "size, log offer price and year (all known before the IPO opens). "
              "Evaluated with accuracy, precision, recall, F1 and confusion matrices.")
    h2(doc, "8.6 Listing Gain Prediction (Regression)")
    para(doc, "Compared Linear Regression, Random Forest, Gradient Boosting, XGBoost "
              "and Huber robust regression over four feature sets (20+ feature × model "
              "combinations in an experiment campaign). The winner is a Huber "
              "regressor on log-transformed features, tuned with GridSearchCV.")
    h2(doc, "8.7 Explainability")
    para(doc, "SHAP values explain each prediction: for the tier classifier they show "
              "which pre-IPO feature pushed the prediction toward Low/Medium/High, and "
              "for the gain model they break the prediction into baseline plus "
              "per-feature contributions.")
    h2(doc, "8.8 Deployment")
    para(doc, "The winning models are cached to disk (models/ipo_models.joblib) and "
              "served by an interactive Streamlit dashboard plus a terminal predictor.")

    h1(doc, "9. Experiments & Results")
    h2(doc, "9.1 Classification Results (out-of-time)")
    tbl = doc.add_table(rows=1, cols=2)
    for i, h in enumerate(["Model", "Out-of-time accuracy"]):
        tbl.rows[0].cells[i].text = h
    for model, acc in [("Logistic Regression", "~59%"), ("Random Forest", "~61%"),
                       ("SVM", "~62% (best)")]:
        cells = tbl.add_row().cells
        cells[0].text = model
        cells[1].text = acc
    style_table(tbl)
    doc.add_paragraph()
    para(doc, "~62% accuracy on IPOs from 2024-2026 (never seen during training) is "
              "the honest ceiling for three pre-IPO features: an experiment campaign "
              "of 20+ configurations all clustered between 60% and 62%. The balanced "
              "class weights improved recall of the rare High tier considerably.")
    add_figure(doc, "cm_random_forest.png",
               "Figure 4 - Confusion matrix of the Random Forest tier classifier (out-of-time test).")
    h2(doc, "9.2 Regression Results (out-of-time)")
    tbl = doc.add_table(rows=1, cols=3)
    for i, h in enumerate(["Setup", "R-squared", "MAE"]):
        tbl.rows[0].cells[i].text = h
    for setup, r2, mae in [
        ("Linear regression, raw features", "-0.20", "21.5 pts"),
        ("Huber regression + log features (final)", "+0.29", "17.8 pts"),
    ]:
        cells = tbl.add_row().cells
        cells[0].text = setup
        cells[1].text = r2
        cells[2].text = mae
    style_table(tbl)
    doc.add_paragraph()
    para(doc, "The log1p transform was the single biggest win: 2024-2026 IPOs are far "
              "more oversubscribed (median ~26x vs ~8x before 2024), so raw values "
              "broke the linear models. Huber (robust regression) then beat ordinary "
              "least squares because listing gains contain heavy outliers. XGBoost "
              "underperformed on this small dataset - consistent with the literature.")
    add_figure(doc, "feature_importance.png",
               "Figure 5 - Feature importance: QIB demand dominates, matching the literature.")
    h2(doc, "9.3 Key Finding: Covariate Shift")
    callout(doc, "⚠️ An honest, valuable finding",
            "The out-of-time test exposed a genuine market shift: recent IPOs are far "
            "more oversubscribed than historical ones (median ~26x vs ~8x). A random "
            "split hid this; the time-based split reveals it - exactly why honest "
            "evaluation matters.",
            fill=PINK_TINT, accent=PINK, title_color=PINK)
    h2(doc, "9.4 Comparison with Similar Work")
    para(doc, "Comparable GitHub projects report 69-80% accuracy, but on binary "
              "profit/loss targets with random splits. Our 62% is a harder test: three "
              "classes and a time-based split. Directly comparing the numbers without "
              "this context would be misleading.")

    h1(doc, "10. Model Deployment & Dashboard")
    para(doc, "The project ships two working interfaces:")
    h2(doc, "10.1 Terminal Predictor (predict_ipo.py)")
    para(doc, "Asks for an IPO's details and prints the predicted subscription tier "
              "(with probabilities) and the predicted listing gain (with a typical "
              "error range). Models are cached on disk, so it starts instantly.")
    h2(doc, "10.2 Streamlit Dashboard (streamlit run app.py)")
    bullet(doc, "Predictor tab: two-stage form (pre-IPO facts, then subscription numbers), an 'Upcoming IPO' mode for IPOs still in bidding, auto-fill from any of the 648 real IPOs, and Actual vs Predicted comparison.")
    bullet(doc, "'Why this prediction?' - SHAP explanations for every forecast.")
    bullet(doc, "Data & EDA tab: interactive 3D playground (year-by-year animation, click-to-highlight, predicted-vs-actual view, 3D gain terrain), subscription donut and Recent IPOs table.")
    bullet(doc, "Model Performance tab: confusion matrices, feature importance and score tables.")
    bullet(doc, "Polish: ticker tape, KPI strip, three colour themes, CSS-3D widget, cursor glow and card hover animations.")
    h2(doc, "10.3 Performance")
    para(doc, "Models are trained once and cached (models/ipo_models.joblib); the "
              "dashboard loads them in ~0.05s instead of retraining for ~15s. They "
              "auto-retrain only when the cleaned dataset changes.")

    h1(doc, "11. Future Scope")
    bullet(doc, "Add grey-market premium (GMP) data - the single strongest known predictor of listing gain.")
    bullet(doc, "Add sector/industry and market context (Nifty trend, market cap, P/E).")
    bullet(doc, "Grow the dataset with 2026+ IPOs and re-tune the classifier once richer features exist.")
    bullet(doc, "Deploy the dashboard online (Streamlit Community Cloud / Hugging Face Spaces) for a shareable live demo.")
    bullet(doc, "Add what-if sliders, an IPO watchlist tracker and a guess-the-tier game to the dashboard.")

    h1(doc, "12. Conclusion")
    para(doc, "This project delivers a complete, honest machine-learning pipeline for "
              "Indian IPO prediction. The subscription-tier classifier reaches ~62% "
              "accuracy on never-seen IPOs, and the listing-gain regressor improves "
              "from a negative out-of-time R-squared to +0.29 through log "
              "transformation and robust regression. The out-of-time evaluation "
              "revealed a real market shift (covariate shift) that random splitting "
              "would have hidden - a finding with genuine value for the report. "
              "Finally, the deployed Streamlit dashboard makes both predictions "
              "usable by a non-technical investor, complete with explanations of why "
              "each prediction was made. The result replaces grey-market rumours with "
              "data-driven signals.")

    h1(doc, "13. References")
    refs = [
        ("Kaggle - IPO Data India 2010-2026",
         "https://www.kaggle.com/datasets/karanammithul/ipo-data-india-2010-2025"),
        ("Maneetss, IPO-prediction-model (GitHub)", None),
        ("kimkmathews, IPO-PerformanceAnalysis (GitHub)", None),
        ("saqibsafdar11, Predicting-Listing-Gains-in-the-Indian-IPO-Market (GitHub)", None),
        ("mborhi, IPO-Underpricing (GitHub)", None),
        ("'Prediction of IPO Subscription - A Logistic Regression Model' (ResearchGate)", None),
        ("Sonsare et al. (2023), IPO underperformance prediction using ANN.", None),
        ("arXiv 2412.16174 - multi-modal ML + NLP approach to IPO success prediction.", None),
        ("Chittorgarh / Zerodha IPO pages - live subscription and GMP reference.",
         "https://www.chittorgarh.com/report/ipo-in-india-list-main-board-sme/82/"),
    ]
    for i, (label, url) in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.add_run(f"[{i}] ")
        if url:
            add_hyperlink(p, url, label)
        else:
            p.add_run(label)

    add_footer(doc)
    doc.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
