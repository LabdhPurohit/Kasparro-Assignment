import json
import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown
from weasyprint import HTML
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Filenames
INSIGHTS_JSON = "insights.json"
COMBINED_CSV = "combined_app_dataset.csv"  
OUT_MD = "executive_report.md"
OUT_HTML = "executive_report.html"
OUT_PDF = "executive_report.pdf"
CHART_DIR = "report_charts"

os.makedirs(CHART_DIR, exist_ok=True)

# ---------- Load insights ----------
with open(INSIGHTS_JSON, "r", encoding="utf-8") as f:
    insights_obj = json.load(f)

numeric = insights_obj.get("numeric", {})
summaries = insights_obj.get("summaries", [])
charts_meta = insights_obj.get("charts", [])

# ---------- Helpers ----------
def fname_safe(s):
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in s).strip().replace(" ", "_")[:120]

def fmt_int(v):
    try:
        return f"{int(v):,}"
    except:
        return str(v)

def fmt_money(v):
    try:
        return f"${v:,.2f}"
    except:
        return str(v)

saved_charts = []

# ---------- Generate Charts ----------
for ch in charts_meta:
    try:
        typ = ch.get("type", "").lower()
        title = ch.get("title", "chart")
        data = ch.get("data", {})
        fname = os.path.join(CHART_DIR, fname_safe(title) + ".png")

        if typ == "bar":
            keys = list(data.keys())
            values = [data[k] for k in keys]
            plt.figure(figsize=(10,6))
            y_pos = np.arange(len(keys))
            plt.barh(y_pos, values, align='center')
            plt.yticks(y_pos, keys)
            plt.gca().invert_yaxis()
            plt.title(title)
            plt.xlabel("Value")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            saved_charts.append(fname)

        elif typ == "comparison":
            keys = list(data.keys())
            values = [data[k] for k in keys]
            plt.figure(figsize=(6,4))
            plt.bar(keys, values)
            plt.title(title)
            plt.ylabel("Avg Installs")
            for i,v in enumerate(values):
                plt.text(i, v*1.02 if v>=0 else v, f"{int(round(v))}", ha='center')
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            saved_charts.append(fname)

        elif typ == "scatter":
            corr = ch.get("correlation", None)
            if os.path.exists(COMBINED_CSV):
                df = pd.read_csv(COMBINED_CSV)
                df['Rating'] = pd.to_numeric(df.get('Rating', pd.Series()), errors='coerce')
                df['Installs'] = pd.to_numeric(df.get('Installs', pd.Series()), errors='coerce')
                df = df.dropna(subset=['Rating','Installs'])
                sample = df.sample(n=min(len(df), 2000), random_state=42) if len(df)>2000 else df
                plt.figure(figsize=(8,5))
                plt.scatter(sample['Rating'], sample['Installs'], alpha=0.35, s=10)
                plt.yscale('log')
                plt.title(f"{title} (sample)")
                plt.xlabel("Rating")
                plt.ylabel("Installs (log scale)")
                if corr is None:
                    corr = float(sample['Rating'].corr(sample['Installs']))
                plt.annotate(f"corr = {round(corr,3)}", xy=(0.95,0.05), xycoords='axes fraction', ha='right', fontsize=9, bbox=dict(boxstyle="round", fc="w"))
                plt.tight_layout()
                plt.savefig(fname, dpi=150)
                plt.close()
                saved_charts.append(fname)
            else:
                plt.figure(figsize=(6,3))
                plt.text(0.5, 0.5, f"Correlation: {round(corr,3)}", ha='center', va='center', fontsize=14)
                plt.axis('off')
                plt.title(title)
                plt.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close()
                saved_charts.append(fname)

    except Exception as e:
        print("Warning: failed to draw chart", title, ":", e)

# ---------- Prepare Data ----------
top_installs = {k: fmt_int(v) for k,v in numeric.get("top_categories_installs", {}).items()}
top_revenue = {k: fmt_money(v) for k,v in numeric.get("top_categories_revenue", {}).items()}
price_downloads = numeric.get("price_vs_downloads", {})
rating_corr = numeric.get("rating_installs_corr", None)

# ---------- Template ----------
env = Environment(
    loader=FileSystemLoader(searchpath="./"),
    autoescape=select_autoescape()
)
env.filters['basename'] = lambda p: os.path.basename(p)

md_template = """
# Executive Market Intelligence Report

**Generated:** {{ generated_time }}

---

## Executive Summary

{% if summaries %}
{% for s in summaries %}
{{ s }}

{% endfor %}
{% else %}
_No AI summaries available._
{% endif %}

---

## Numeric Insights

### Top Categories by Installs
{% for k,v in top_installs.items() %}
- **{{ k }}** — {{ v }}
{% endfor %}

### Top Categories by Revenue
{% for k,v in top_revenue.items() %}
- **{{ k }}** — {{ v }}
{% endfor %}

### Price vs Downloads
- Avg installs (Free): {{ price_downloads.get('avg_free_installs') | default('N/A') | round(0) }}
- Avg installs (Paid): {{ price_downloads.get('avg_paid_installs') | default('N/A') | round(0) }}

### Rating vs Installs
- Correlation: {{ rating_corr }}

---

## Charts

{% if charts %}
{% for c in charts %}
### Chart {{ loop.index }}: {{ c | basename }}
![chart{{ loop.index }}]({{ c }})
{% endfor %}
{% else %}
_No charts available._
{% endif %}

---

"""

template = env.from_string(md_template)

md = template.render(
    generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    summaries=summaries,
    top_installs=top_installs,
    top_revenue=top_revenue,
    price_downloads=price_downloads,
    rating_corr=rating_corr,
    charts=saved_charts,
    meta_rows=insights_obj.get("meta", {}).get("source_rows", "unknown")
)

# ---------- Save Outputs ---------
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

html_body = markdown.markdown(md, extensions=['fenced_code','tables'])
html_page = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Executive Market Intelligence Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 36px; color: #111; }}
    h1,h2,h3 {{ color: #0b57d0; }}
    img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; padding: 4px; background:#fff; }}
    pre {{ background:#f6f8fa; padding:10px; overflow-x:auto; }}
    blockquote {{ color:#666; border-left:3px solid #eee; padding-left:10px; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_page)

print("Converting HTML to PDF...")
HTML(OUT_HTML).write_pdf(OUT_PDF)
print("✅ Saved:", OUT_MD, OUT_HTML, OUT_PDF)
print("Charts saved in:", CHART_DIR)
