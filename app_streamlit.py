import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import plotly.express as px
from openai import OpenAI
import os, json
from datetime import datetime
from weasyprint import HTML

CSV_PATH = "combined_app_dataset.csv"
INSIGHTS_JSON = "insights.json"
REPORT_PDF = "executive_report.pdf"
CHART_DIR = "pdf_charts"
os.makedirs(CHART_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="Kasparro — App Intelligence")

# ---------------- Load Data ----------------
@st.cache_data
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in ["Installs","Rating","Price","Reviews","Match_Score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()


# ---------------- Insights ----------------
def compute_insights(df):
    df = df.dropna(subset=["Category"])
    top_by_installs = df.groupby("Category")["Installs"].sum().sort_values(ascending=False).head(5)
    df["Revenue"] = df["Price"].fillna(0) * df["Installs"].fillna(0)
    top_by_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(5)
    free_apps = df[df["Price"]==0]["Installs"].mean()
    paid_apps = df[df["Price"]>0]["Installs"].mean()
    corr_df = df.dropna(subset=["Rating","Installs"])
    correlation = corr_df["Rating"].corr(corr_df["Installs"])
    return {
        "top_installs": top_by_installs,
        "top_revenue": top_by_revenue,
        "free_avg": free_apps,
        "paid_avg": paid_apps,
        "correlation": correlation
    }

def generate_summaries(insights):
    summaries = []
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY","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"))
        prompts = [
            f"Category {insights['top_installs'].index[0]} leads installs with {insights['top_installs'].iloc[0]:,}.",
            f"Category {insights['top_revenue'].index[0]} drives revenue ${insights['top_revenue'].iloc[0]:,.0f}.",
            f"Free apps average {insights['free_avg']:.0f} installs, paid apps average {insights['paid_avg']:.0f}.",
            f"Correlation between rating and installs: {insights['correlation']:.2f}."
        ]
        for p in prompts:
            comp = client.chat.completions.create(
                model="x-ai/grok-4-fast:free",
                messages=[{"role":"system","content":"Write short, clear executive summaries."},
                          {"role":"user","content":p}]
            )
            summaries.append(comp.choices[0].message.content)
    except Exception:
        # fallback
        summaries = [
            f"Most installs: {insights['top_installs'].index[0]}",
            f"Most revenue: {insights['top_revenue'].index[0]}",
            f"Free vs Paid: {insights['free_avg']:.0f} vs {insights['paid_avg']:.0f}",
            f"Rating-install correlation: {insights['correlation']:.2f}"
        ]
    return summaries

def save_pdf(summaries, df):
    charts = []

    # Plotly: Top Installs
    inst = df.groupby("Category")["Installs"].sum().reset_index().sort_values("Installs", ascending=False).head(5)
    fig1 = px.bar(inst, x="Category", y="Installs", title="Top Categories by Installs")
    fname1 = os.path.join(CHART_DIR, "top_installs.png")
    fig1.write_image(fname1)
    charts.append(fname1)

    # Plotly: Top Revenue
    df["Revenue"] = df["Price"].fillna(0) * df["Installs"].fillna(0)
    rev = df.groupby("Category")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False).head(5)
    fig2 = px.bar(rev, x="Category", y="Revenue", title="Top Categories by Revenue", color_discrete_sequence=["orange"])
    fname2 = os.path.join(CHART_DIR, "top_revenue.png")
    fig2.write_image(fname2)
    charts.append(fname2)

    # Plotly: Scatter
    scatter_df = df.dropna(subset=["Rating","Installs"])
    if not scatter_df.empty:
        fig3 = px.scatter(scatter_df, x="Rating", y="Installs", size="Reviews", hover_name="App",
                          title="Rating vs Installs", log_y=True)
        fname3 = os.path.join(CHART_DIR, "scatter.png")
        fig3.write_image(fname3)
        charts.append(fname3)

    # HTML for PDF
    summary_html = "<ul>" + "".join(f"<li>{s}</li>" for s in summaries) + "</ul>"
    chart_imgs = "".join(f"<h3>{os.path.basename(c)}</h3><img src='{c}' style='max-width:600px;'/>" for c in charts)
    html_page = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1,h2,h3 {{ color: #0b57d0; }}
        img {{ margin: 20px 0; border: 1px solid #ddd; padding: 4px; }}
      </style>
    </head>
    <body>
      <h1>Executive Report</h1>
      <h2>Summaries</h2>
      {summary_html}
      <h2>Charts</h2>
      {chart_imgs}
    </body>
    </html>
    """
    HTML(string=html_page, base_url=".").write_pdf(REPORT_PDF)
    return REPORT_PDF

# ---------------- UI ----------------
tab1, tab2, tab3 = st.tabs(["🔎 Query Apps","📈 Insights","📄 Report"])


# --- Tab1 Query ---
with tab1:
    st.title("Kasparro — App Query & Compare Interface")
    st.markdown("Search Google Play apps, compare Apple matches, filter and export results.")
    st.sidebar.header("Filters")

    genre_opts = sorted(df["Genre"].dropna().unique().tolist()) if "Genre" in df.columns else []
    sel_genre = st.sidebar.multiselect("Genre", options=genre_opts)
    cat_opts = sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else []
    sel_cat = st.sidebar.multiselect("Category", options=cat_opts)
    source_opts = sorted(df["Source"].dropna().unique().tolist()) if "Source" in df.columns else []
    sel_source = st.sidebar.multiselect("Source", options=source_opts)
    min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0,5.0), 0.1)
    price_min = st.sidebar.number_input("Price min", value=0.0)
    price_max = st.sidebar.number_input("Price max", value=100.0)

    working = df.copy()
    if sel_genre: working = working[working["Genre"].isin(sel_genre)]
    if sel_cat: working = working[working["Category"].isin(sel_cat)]
    if sel_source: working = working[working["Source"].isin(sel_source)]
    working = working[(working["Rating"].fillna(0) >= min_rating) & (working["Rating"].fillna(0) <= max_rating)]
    if "Price" in working.columns:
        working = working[(working["Price"].fillna(0) >= price_min) & (working["Price"].fillna(0) <= price_max)]

    st.write(f"Rows: {len(working)} — Showing first 200")
    st.dataframe(working.head(200))

    # ---------------- Charts ----------------
    st.subheader("📊 Visualizations on Filtered Data")

    if not working.empty:
        # Rating distribution
        if "Rating" in working.columns:
            fig_rating = px.histogram(working, x="Rating", nbins=20, title="Rating Distribution", marginal="box")
            st.plotly_chart(fig_rating, use_container_width=True)

        # Rating vs Installs scatter (log scale)
        if "Rating" in working.columns and "Installs" in working.columns:
            fig_scatter = px.scatter(
                working, x="Rating", y="Installs",
                size="Reviews" if "Reviews" in working.columns else None,
                hover_name="App",
                log_y=True,
                title="Rating vs Installs (log scale)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Genre distribution (Pie)
        if "Genre" in working.columns:
            genre_counts = working["Genre"].value_counts().reset_index()
            genre_counts.columns = ["Genre","Count"]
            fig_pie = px.pie(genre_counts, names="Genre", values="Count", title="Genre Distribution", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

# --- Tab2 Insights ---
with tab2:
    st.title("📈 Insights Dashboard")
    insights = compute_insights(df)
    summaries = generate_summaries(insights)

    st.subheader("Executive Summaries")
    for s in summaries: st.write("✔️", s)

    col1,col2,col3 = st.columns(3)
    col1.metric("Free avg installs", f"{insights['free_avg']:.0f}")
    col2.metric("Paid avg installs", f"{insights['paid_avg']:.0f}")
    col3.metric("Rating-Install Corr", f"{insights['correlation']:.2f}")

    st.subheader("Charts")
    st.plotly_chart(px.bar(insights["top_installs"].reset_index(), x="Category", y="Installs", title="Top Categories by Installs"), use_container_width=True)
    st.plotly_chart(px.bar(insights["top_revenue"].reset_index(), x="Category", y="Revenue", title="Top Categories by Revenue", color_discrete_sequence=["orange"]), use_container_width=True)
    scatter_df = df.dropna(subset=["Rating","Installs"])
    if not scatter_df.empty:
        st.plotly_chart(px.scatter(scatter_df, x="Rating", y="Installs", size="Reviews", hover_name="App", title="Rating vs Installs", log_y=True), use_container_width=True)

    with open(INSIGHTS_JSON,"w") as f: json.dump({"summaries":summaries},f,indent=2)

# --- Tab3 Report ---
with tab3:
    st.title("📄 Download Report")
    if os.path.exists(INSIGHTS_JSON):
        with open(INSIGHTS_JSON) as f: summaries=json.load(f)["summaries"]
        if st.button("Generate PDF"):
            pdf = save_pdf(summaries, df)
            with open(pdf,"rb") as f:
                st.download_button("Download PDF", data=f, file_name="executive_report.pdf", mime="application/pdf")
    else:
        st.info("Generate insights first in the Insights tab.")
