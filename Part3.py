import pandas as pd
import numpy as np
import json
from openai import OpenAI

# ---------- Load Data ----------
df = pd.read_csv("combined_app_dataset.csv")

# Clean key numeric columns
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

# Drop rows with no category
df = df.dropna(subset=["Category"])

# ---------- Basic Insights ----------
# Top categories by Installs
top_by_installs = df.groupby("Category")["Installs"].sum().sort_values(ascending=False).head(5)

# Top categories by Revenue
df["Revenue"] = df["Price"] * df["Installs"].fillna(0)
top_by_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(5)

# Free vs Paid installs
free_apps = df[df["Price"] == 0]["Installs"].mean()
paid_apps = df[df["Price"] > 0]["Installs"].mean()

# Rating vs Installs correlation
corr_df = df.dropna(subset=["Rating","Installs"])
correlation = corr_df["Rating"].corr(corr_df["Installs"])

# ---------- Confidence Score ----------
def confidence_score(n, variance):
    if n == 0:
        return 0
    return min(0.99, (1 - np.exp(-n/50)) * (1 / (1 + variance)))

# ---------- LLM Client ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def generate_text_insight(prompt):
    completion = client.chat.completions.create(
        model="x-ai/grok-4-fast:free",
        messages=[
            {"role": "system", "content": "You are a data insights assistant. Write short, clear insights for executives."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# ---------- LLM Summaries ----------
summaries = []
summaries.append(generate_text_insight(f"Category {top_by_installs.index[0]} has the most installs: {top_by_installs.iloc[0]}."))
summaries.append(generate_text_insight(f"Category {top_by_revenue.index[0]} has the most revenue: {top_by_revenue.iloc[0]:.2f}."))
summaries.append(generate_text_insight(f"Free apps average {free_apps:.0f} installs, while paid apps average {paid_apps:.0f}."))
summaries.append(generate_text_insight(f"The correlation between rating and installs is {correlation:.2f}."))

# ---------- Chart Data ----------
chart_suggestions = {
    "charts": [
        {
            "type": "bar",
            "title": "Top Categories by Installs",
            "data": top_by_installs.to_dict()
        },
        {
            "type": "bar",
            "title": "Top Categories by Revenue",
            "data": top_by_revenue.to_dict()
        },
        {
            "type": "comparison",
            "title": "Free vs Paid Installs",
            "data": {"Free": free_apps, "Paid": paid_apps}
        },
        {
            "type": "scatter",
            "title": "Rating vs Installs Correlation",
            "correlation": correlation
        }
    ]
}

# ---------- Final Insights JSON ----------
insights = {
    "numeric": {
        "top_categories_installs": top_by_installs.to_dict(),
        "top_categories_revenue": top_by_revenue.to_dict(),
        "price_vs_downloads": {
            "avg_free_installs": free_apps,
            "avg_paid_installs": paid_apps
        },
        "rating_installs_corr": correlation
    },
    "summaries": summaries,   # NEW: human-readable insights
    "charts": chart_suggestions["charts"]  # NEW: chart metadata
}

with open("insights.json", "w") as f:
    json.dump(insights, f, indent=2)

print("✅ insights.json saved with numeric + summaries + charts")
