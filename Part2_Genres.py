import http.client, json, urllib.parse, time, re, os
import pandas as pd
from rapidfuzz import fuzz
from openai import OpenAI

# ------------------ CONFIG ------------------
API_HOST = "appstore-scrapper-api.p.rapidapi.com"
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") or "714ca93dadmshe1f5a322b319523p1afd5cjsn40ef8adb09d6"
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY") or "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"

headers = {
    'x-rapidapi-key': RAPIDAPI_KEY,
    'x-rapidapi-host': API_HOST
}

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

# ------------------ API ------------------
def search_apps(query, country="us", num=50, lang="en"):
    safe_query = urllib.parse.quote(query)
    path = f"/v1/app-store-api/search?num={num}&lang={lang}&query={safe_query}&country={country}"
    conn = http.client.HTTPSConnection(API_HOST, timeout=30)
    conn.request("GET", path, headers=headers)
    res = conn.getresponse()
    data = res.read()
    try:
        result = json.loads(data.decode("utf-8"))
        if isinstance(result, dict) and "apps" in result:
            return result["apps"]
        elif isinstance(result, list):
            return result
        else:
            return []
    except Exception as e:
        print("❌ JSON decode error:", e)
        return []

# ------------------ Helpers ------------------
GENERIC_WORDS = set([
    "coloring", "book", "editor", "photo", "video", "game", "free", "app",
    "puzzle", "kids", "camera", "guide", "pro", "hd", "3d", "lite", "plus",
    "themes", "widget", "launcher", "live", "cool"
])

def normalize(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_keywords(name):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return [w for w in words if w not in GENERIC_WORDS and len(w) > 2]

def best_match_strict(gname, apps):
    gnorm = normalize(gname)
    gkeywords = extract_keywords(gnorm)
    scored = []
    for app in apps:
        aname = app.get("title", "")
        anorm = normalize(aname)
        akeywords = extract_keywords(anorm)
        overlap = set(gkeywords) & set(akeywords)
        score = fuzz.token_set_ratio(gnorm, anorm)
        if not overlap and gkeywords:
            score *= 0.5
        reviews = app.get("reviews") or 0
        scored.append((aname, score, reviews, app, overlap))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored

def ai_rerank_index(gname, candidates_scored):
    """Ask AI to pick candidate index (1..N) or None."""
    lines = []
    for idx, (title, score, reviews, _, overlap) in enumerate(candidates_scored[:5], 1):
        tokens = list(overlap) if overlap else []
        lines.append(f"{idx}. {title} | score={round(score,2)} | reviews={reviews} | overlap={tokens}")
    prompt = f"""
Google app name: "{gname}"
Apple candidates:
{chr(10).join(lines)}

Which one is the same app? Reply ONLY with the number (1..5) or 'None'.
"""
    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.0
        )
        text = resp.choices[0].message.content.strip()
        print("🤖 AI raw:", text)
        if re.search(r"\bNone\b", text, re.IGNORECASE):
            return None
        m = re.search(r"(\d+)", text)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= len(candidates_scored[:5]):
                return idx
        return None
    except Exception as e:
        print("⚠️ AI error:", e)
        return None

# ------------------ Main ------------------
def merge_google_apple(google_csv="clean_google_play_data.csv", out_csv="combined_app_dataset2.csv"):
    google_df = pd.read_csv(google_csv)

    # 1. Build Apple dataset by genres
    genres = sorted(set(google_df["Genre"].dropna().unique().tolist()))
    apple_apps = []
    for g in genres:
        print(f"\n📂 Fetching Apple apps for genre: {g}")
        apps = search_apps(g, num=50)
        for app in apps:
            app["Genre_query"] = g
        apple_apps.extend(apps)
        time.sleep(1)
    apple_df = pd.DataFrame(apple_apps)
    print(f"\n✅ Collected {len(apple_df)} Apple apps across {len(genres)} genres")

    # 2. Matching Google apps against Apple apps
    merged_rows = []
    for i, row in google_df.iterrows():
        gname = row["App"]
        ggenre = row["Genre"]
        print("="*80)
        print(f"🔍 Google App: '{gname}' | Genre: {ggenre}")

        # restrict Apple candidates to same genre
        candidates = apple_df[apple_df["Genre_query"] == ggenre].to_dict(orient="records")
        if not candidates:
            print("⚠️ No Apple results in this genre")
            merged = row.to_dict()
            merged.update({"App_apple": None, "Source": "google-only", "Match_Score": 0})
            merged_rows.append(merged)
            continue

        scored = best_match_strict(gname, candidates)

        for rank, (aname, score, reviews, _, overlap) in enumerate(scored, 1):
            print(f"  Candidate {rank}: '{aname}' → Score {round(score,2)}, Reviews {reviews}, Overlap {overlap}")

        best_aname, best_score, _, best_app, best_overlap = scored[0]

        accepted, chosen_app, chosen_score = False, None, best_score

        if best_score >= 85:
            accepted, chosen_app = True, best_app
            print(f"✅ Local accept: {best_aname} ({best_score})")
        elif 70 <= best_score < 85 and not best_overlap:
            print("⚠️ Borderline → sending to AI")
            ai_idx = ai_rerank_index(gname, scored)
            if ai_idx:
                cand = scored[ai_idx-1]
                if cand[1] >= 75 or (cand[1] >= 60 and cand[4]):
                    accepted, chosen_app, chosen_score = True, cand[3], cand[1]
                    print(f"🤖 AI accepted: {cand[0]} ({cand[1]})")
        elif 70 <= best_score < 85 and best_overlap:
            accepted, chosen_app = True, best_app
            print(f"✅ Local accept with overlap: {best_aname} ({best_score})")
        else:
            print("❌ Low score, no confident match")

        merged = row.to_dict()
        if accepted and chosen_app:
            merged.update({
                "App_apple": chosen_app.get("title"),
                "Rating_apple": chosen_app.get("score"),
                "Reviews_apple": chosen_app.get("reviews"),
                "Price_apple": chosen_app.get("price"),
                "Developer_apple": chosen_app.get("developer"),
                "Match_Score": chosen_score,
                "Source": "both"
            })
        else:
            merged.update({
                "App_apple": None,
                "Rating_apple": None,
                "Reviews_apple": None,
                "Price_apple": None,
                "Developer_apple": None,
                "Match_Score": best_score,
                "Source": "google-only"
            })

        merged_rows.append(merged)
        if (i+1) % 10 == 0:
            print(f"Progress: {i+1} apps matched")

    df = pd.DataFrame(merged_rows)
    df.to_csv(out_csv, index=False)
    print("\n✅ Finished merging!")
    print(f"Saved to {out_csv} with {len(df)} rows")
    return df

# ------------------ Run ------------------
if __name__ == "__main__":
    df = merge_google_apple()
    print("\nSample merged rows:")
    print(df[["App","App_apple","Rating","Rating_apple","Match_Score","Source"]].head(10).to_string(index=False))
