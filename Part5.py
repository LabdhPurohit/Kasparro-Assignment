import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import requests

# --------------------
# CONFIG
# --------------------
DATA_FILE = Path("data.xlsx")
OUT_DIR = Path("phase5_outputs")
OUT_DIR.mkdir(exist_ok=True)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")  # or set string here
OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"  # change if needed
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# --------------------
# HELPERS
# --------------------
def safe_div(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(b == 0, np.nan, a / b)
    return out

def print_head(msg):
    print("="*6 + " " + msg + " " + "="*6)

# --------------------
# 1) Load & normalize
# --------------------
print_head("Loading dataset")
if not DATA_FILE.exists():
    raise FileNotFoundError(f"{DATA_FILE} not found. Put the Excel file in the script folder.")

# Read first sheet
df = pd.read_excel(DATA_FILE, sheet_name=0)

# Lowercase and replace spaces with underscores
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("Columns found:", df.columns.tolist())

# Rename some columns to canonical names if needed (map common variants)
rename_map = {
    "seo_category": "category",
    "monthly_search_volume": "search_volume",
    "conversion_rate": "seo_conv_rate",
    "avg_position": "avg_position",  # already same
    "spend_usd": "spend_usd",
    "revenue_usd": "revenue_usd",
    "first_purchase": "first_purchase",
    "repeat_purchase": "repeat_purchase",
    "installs": "installs",
    "signups": "signups",
    "impressions": "impressions",
    "clicks": "clicks"
}
# Only rename keys that exist in df
actual_rename = {k: v for k, v in rename_map.items() if k in df.columns and k != v}
if actual_rename:
    df = df.rename(columns=actual_rename)
    print("Applied rename map:", actual_rename)

# Ensure numeric types for expected numeric columns
num_cols = [
    "spend_usd","impressions","clicks","revenue_usd",
    "installs","signups","first_purchase","repeat_purchase",
    "search_volume","avg_position","seo_conv_rate"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# --------------------
# 2) Per-row (campaign) KPIs
# --------------------
print_head("Computing campaign-level KPIs")

# CTR, CPC
if "clicks" in df.columns and "impressions" in df.columns:
    df["ctr"] = safe_div(df["clicks"], df["impressions"])
else:
    df["ctr"] = np.nan

if "spend_usd" in df.columns and "clicks" in df.columns:
    df["cpc"] = safe_div(df["spend_usd"], df["clicks"])
else:
    df["cpc"] = np.nan


if "first_purchase" in df.columns:
    df["cac"] = safe_div(df.get("spend_usd", np.nan), df.get("first_purchase", np.nan))
elif "signups" in df.columns:
    df["cac"] = safe_div(df.get("spend_usd", np.nan), df.get("signups", np.nan))
else:
    df["cac"] = np.nan

# ROAS
if "revenue_usd" in df.columns and "spend_usd" in df.columns:
    df["roas"] = safe_div(df["revenue_usd"], df["spend_usd"])
else:
    df["roas"] = np.nan

# Funnel ratios
if "installs" in df.columns and "signups" in df.columns:
    df["install_to_signup_rate"] = safe_div(df["signups"], df["installs"])
else:
    df["install_to_signup_rate"] = np.nan

if "signups" in df.columns and "first_purchase" in df.columns:
    df["signup_to_first_rate"] = safe_div(df["first_purchase"], df["signups"])
else:
    df["signup_to_first_rate"] = np.nan

if "first_purchase" in df.columns and "repeat_purchase" in df.columns:
    df["first_to_repeat_rate"] = safe_div(df["repeat_purchase"], df["first_purchase"])
else:
    df["first_to_repeat_rate"] = np.nan

df["repeat_rate"] = df.get("first_to_repeat_rate", np.nan)  # alias

# Save cleaned campaign-level CSV
cleaned_csv = OUT_DIR / "cleaned_campaigns.csv"
df.to_csv(cleaned_csv, index=False)
print("Cleaned campaign data saved to:", cleaned_csv)

# --------------------
# 3) Category-level SEO opportunity
# --------------------
print_head("Computing SEO opportunity by category")

if all(c in df.columns for c in ["category", "search_volume", "avg_position", "seo_conv_rate"]):
    # normalize avg_position so that lower (1) is better -> normalized score 0..1 where higher is better
    pos = df["avg_position"].astype(float)
    # handle constant pos case
    pos_min = np.nanmin(pos)
    pos_max = np.nanmax(pos)
    if np.isfinite(pos_min) and np.isfinite(pos_max) and pos_max != pos_min:
        pos_norm = (pos_max - pos) / (pos_max - pos_min)
    else:
        pos_norm = np.where(np.isfinite(pos), 1.0, 0.0)

    # per-row seo opportunity
    df["seo_opportunity_row"] = df["search_volume"].rank(pct=True) * pos_norm * df["seo_conv_rate"].fillna(0)

    # aggregate at category level
    seo_cat = (
        df.groupby("category")
          .agg(
              seo_opportunity_score=("seo_opportunity_row", "mean"),
              search_volume=("search_volume", "sum"),
              avg_position=("avg_position", "mean"),
              seo_conv_rate=("seo_conv_rate", "mean"),
              campaigns_count=("campaign_id", "count")
          )
          .reset_index()
          .sort_values("seo_opportunity_score", ascending=False)
    )
    seo_cat.to_csv(OUT_DIR / "seo_category_scores.csv", index=False)
    print("SEO category scores saved to:", OUT_DIR / "seo_category_scores.csv")
else:
    seo_cat = pd.DataFrame()
    print("Skipping SEO opportunity: required columns missing (category, search_volume, avg_position, seo_conv_rate).")

# --------------------
# 4) Insights JSON
# --------------------
print_head("Preparing insights.json")
insights = {}

# Median metrics
insights["median_cac"] = float(np.nanmedian(df["cac"])) if "cac" in df.columns else None
insights["median_roas"] = float(np.nanmedian(df["roas"])) if "roas" in df.columns else None
insights["median_repeat_rate"] = float(np.nanmedian(df["repeat_rate"])) if "repeat_rate" in df.columns else None

# Top campaigns by ROAS and lowest CAC
insights["top_roas_campaigns"] = []
insights["best_cac_campaigns"] = []
if "campaign_id" in df.columns:
    roas_sorted = df.sort_values("roas", ascending=False).head(10)
    insights["top_roas_campaigns"] = roas_sorted[["campaign_id","roas","spend_usd","revenue_usd"]].fillna("").to_dict(orient="records")
    cac_sorted = df.sort_values("cac", ascending=True).head(10)
    insights["best_cac_campaigns"] = cac_sorted[["campaign_id","cac","spend_usd"]].fillna("").to_dict(orient="records")

# Add SEO top categories (if computed)
insights["top_seo_categories"] = []
if not seo_cat.empty:
    insights["top_seo_categories"] = seo_cat.head(10).to_dict(orient="records")

# Add summary stats
insights["total_campaigns"] = int(df.shape[0])
insights["generated_at_utc"] = datetime.utcnow().isoformat() + "Z"

# Save insights
ins_path = OUT_DIR / "insights.json"
with open(ins_path, "w") as f:
    json.dump(insights, f, indent=2, default=str)
print("Insights saved to:", ins_path)

# --------------------
# 5) Creative generation (OpenRouter) OR fallback
# --------------------
print_head("Generating creatives (OpenRouter if key provided)")

# Build a short structured prompt using top categories and top campaign signals
top_categories = [c.get("category") for c in insights.get("top_seo_categories", [])][:3]
top_roas = insights.get("top_roas_campaigns", [])[:3]

prompt_summary = {
    "median_cac": insights.get("median_cac"),
    "median_roas": insights.get("median_roas"),
    "median_repeat_rate": insights.get("median_repeat_rate"),
    "top_categories": top_categories,
    "top_roas_examples": top_roas
}

prompt_text = f"""
You are a clever copywriter for a D2C brand. Use the insights below to generate marketing creatives.

INSIGHTS:
{json.dumps(prompt_summary, indent=2)}

TASK:
1) Generate 3 ad headlines (each <=30 characters).
2) Generate 2 SEO meta descriptions (each <=160 characters).
3) Generate 1 product detail paragraph (~60-90 words).
Tone: modern, persuasive, friendly D2C brand voice. Keep outputs JSON-serializable and labeled.

Return only plain text (no markdown) that is easy to parse.
"""

creative_output_text = ""
if OPENROUTER_API_KEY:
    try:
        resp = requests.post(
            OPENROUTER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 800
            },
            timeout=30
        )
        resp.raise_for_status()
        j = resp.json()
        # Try to extract assistant message (format may vary)
        try:
            creative_output_text = j["choices"][0]["message"]["content"]
        except Exception:
            creative_output_text = json.dumps(j, indent=2)
        print("OpenRouter response received.")
    except Exception as e:
        creative_output_text = f"Error calling OpenRouter: {e}\n\nResponse (if any): {getattr(resp, 'text', '')}"
        print("OpenRouter call failed:", e)
else:
    # fallback templated creatives if no key provided
    creative_output_text = ""
    # Use top insights to create basic templated creatives
    headline_base = "Shop Our Bestsellers"
    if top_categories:
        creative_output_text += "Ad Headlines:\n"
        for i, cat in enumerate(top_categories[:3], start=1):
            creative_output_text += f"{i}. {cat} — {headline_base}\n"
        creative_output_text += "\nSEO Meta Descriptions:\n"
        for i, cat in enumerate(top_categories[:2], start=1):
            creative_output_text += f"{i}. Discover top {cat} products — free shipping & easy returns. Shop now.\n"
        creative_output_text += "\nPDP Paragraph:\n"
        creative_output_text += "Our curated bestselling collection combines premium ingredients/design with everyday value. Loved by hundreds of customers for visible results and fast delivery. Try risk-free with our 30-day returns.\n"
    else:
        creative_output_text = ("Ad Headlines:\n1. Shop Our Bestsellers\n2. New Arrivals — Limited Stock\n3. Save 15% Today\n\n"
                                "SEO Meta Descriptions:\n1. Shop top-rated D2C products with fast shipping. Discover deals today.\n2. Explore bestsellers and get special discounts. Free returns.\n\n"
                                "PDP Paragraph:\nOur products are crafted for quality and results. Fast shipping, simple returns, and loved by customers. Try now and see why people come back.\n")

# Save creatives
creatives_path = OUT_DIR / "creatives.txt"
with open(creatives_path, "w") as f:
    f.write(creative_output_text)
print("Creatives saved to:", creatives_path)

# --------------------
# 6) Executive report (markdown)
# --------------------
print_head("Writing executive_report.md")
report_lines = []
report_lines.append(f"# Phase 5 D2C Executive Report\nGenerated: {datetime.utcnow().isoformat()} UTC\n")
report_lines.append("## Topline KPIs\n")
report_lines.append(f"- Total campaigns: {insights.get('total_campaigns')}\n")
report_lines.append(f"- Median CAC: {insights.get('median_cac')}\n")
report_lines.append(f"- Median ROAS: {insights.get('median_roas')}\n")
report_lines.append(f"- Median Repeat Rate: {insights.get('median_repeat_rate')}\n")

if insights.get("top_seo_categories"):
    report_lines.append("\n## Top SEO Categories (by opportunity score)\n")
    for row in insights["top_seo_categories"]:
        report_lines.append(
            f"- {row.get('category')}: score={row.get('seo_opportunity_score'):.4f}, "
            f"search_volume={row.get('search_volume')}, avg_position={row.get('avg_position'):.1f}\n"
        )

report_lines.append("\n## Top ROAS Campaigns (sample)\n")
for c in insights.get("top_roas_campaigns", [])[:5]:
    report_lines.append(f"- {c.get('campaign_id')}: ROAS={c.get('roas')}, spend={c.get('spend_usd')}, revenue={c.get('revenue_usd')}\n")

report_lines.append("\n## AI / Generated Creatives\n")
report_lines.append("\n```\n")
report_lines.append(creative_output_text)
report_lines.append("\n```\n")

report_md_path = OUT_DIR / "executive_report.md"
with open(report_md_path, "w") as f:
    f.write("\n".join(report_lines))

print("Executive report saved to:", report_md_path)
print_head("Done")
print(f"All outputs are in the folder: {OUT_DIR.resolve()}")
