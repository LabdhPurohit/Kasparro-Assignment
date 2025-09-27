import http.client, json, urllib.parse, time, re
import pandas as pd
from rapidfuzz import fuzz
from openai import OpenAI
import os
import sys

# ------------------ CONFIG ------------------
API_HOST = "appstore-scrapper-api.p.rapidapi.com"
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") or "714ca93dadmshe1f5a322b319523p1afd5cjsn40ef8adb09d6"
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY") or "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"

headers = {
    'x-rapidapi-key': RAPIDAPI_KEY,
    'x-rapidapi-host': API_HOST
}

# OpenRouter client wrapper (same pattern you used)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

# ------------------ API Search ------------------
def search_apps(query, country="us", num=5, lang="en"):
    safe_query = urllib.parse.quote(query)
    path = f"/v1/app-store-api/search?num={num}&lang={lang}&query={safe_query}&country={country}"
    conn = http.client.HTTPSConnection(API_HOST, timeout=30)
    conn.request("GET", path, headers=headers)
    res = conn.getresponse()
    data = res.read()
    try:
        result = json.loads(data.decode("utf-8"))
        # handle both shapes: {"apps":[...]} or [ {...}, ... ]
        if isinstance(result, dict) and "apps" in result:
            return result["apps"]
        elif isinstance(result, list):
            return result
        else:
            return []
    except Exception as e:
        print("❌ JSON decode error:", e)
        return []

# ------------------ Matching utilities ------------------
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
    """Return list of candidates: (title, score, reviews, app_obj, overlap_set) sorted desc."""
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
            score = score * 0.5
        reviews = app.get("reviews") or 0
        scored.append((aname, score, reviews, app, overlap))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored

# ------------------ AI re-ranker (returns index OR None) ------------------
def ai_rerank_index(gname, candidates_scored):
    """
    Ask the LLM to pick the index (1..N) of the candidate that is the same app as gname, or 'None'.
    Returns integer index (1-based) or None.
    """
    # Build prompt with structured numbered candidates and useful metadata
    lines = []
    for idx, (title, score, reviews, app_obj, overlap) in enumerate(candidates_scored[:5], start=1):
        tokens = list(overlap) if overlap else []
        lines.append(f"{idx}. Title: {title} | fuzzy_score: {round(score,2)} | reviews: {reviews} | overlap_tokens: {tokens}")

    prompt = f"""You are a precise assistant that must pick the single Apple candidate that represents the same app as the Google app name.
Google app name: "{gname}"

Apple candidates (numbered):
{chr(10).join(lines)}

Instructions:
- Return ONLY a single number (the candidate index 1..{min(5, len(candidates_scored))}) if that candidate is the same app.
- If none of the candidates match, return the word: None
- DO NOT add any other text or explanation. ONLY the number or None.
"""
    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=[
                {"role":"system","content":"You are a short-answer classifier. Reply with only the index number or None."},
                {"role":"user","content":prompt}
            ],
            max_tokens=32,
            temperature=0.0
        )
        text = resp.choices[0].message.content.strip()
        print("🤖 Raw AI reply:", repr(text))
        # parse integer or 'None'
        m = re.search(r"\b(None|null)\b", text, flags=re.IGNORECASE)
        if m:
            return None
        m2 = re.search(r"(\d+)", text)
        if m2:
            idx = int(m2.group(1))
            if 1 <= idx <= min(5, len(candidates_scored)):
                return idx
        print("⚠️ AI reply couldn't be parsed as valid index or None.")
        return None
    except Exception as e:
        print("⚠️ AI error:", e)
        return None

# ------------------ Merge main ------------------
def merge_google_apple(google_csv="clean_google_play_data.csv", out_csv="combined_app_dataset.csv"):
    google_df = pd.read_csv(google_csv)
    merged_rows = []

    for i, row in google_df.iterrows():
        gname = row["App"]
        print("="*80)
        print(f"🔍 Google App: '{gname}'")

        apps = search_apps(gname, num=5)
        if not apps:
            print("⚠️ No Apple results")
            merged = row.to_dict()
            merged.update({"App_apple": None, "Source": "google-only", "Match_Score": 0})
            merged_rows.append(merged)
            continue

        scored = best_match_strict(gname, apps)

        for rank, (aname, score, reviews, _, overlap) in enumerate(scored, 1):
            print(f"  Candidate {rank}: '{aname}' → Score {round(score,2)}, Reviews {reviews}, Overlap {overlap}")

        best_aname, best_score, best_reviews, best_app_obj, best_overlap = scored[0]
        # Decision rules:
        # 1) If best_score >= 85 -> accept immediately
        if best_score >= 85:
            chosen_app = best_app_obj
            chosen_score = best_score
            accepted = True
            print(f"✅ Local accept (score >= 85): {best_aname} ({round(best_score,2)})")
        else:
            # 2) If 70 <= best_score < 85:
            #    if overlap present -> accept (keyword overlap indicates stronger match)
            #    else call AI re-ranker
            accepted = False
            chosen_app = None
            chosen_score = best_score
            if 70 <= best_score < 85:
                if best_overlap:
                    accepted = True
                    chosen_app = best_app_obj
                    print(f"✅ Local accept (70-85 + overlap): {best_aname} ({round(best_score,2)})")
                else:
                    print("⚠️ Borderline (70-85) with NO keyword overlap -> invoking AI re-ranker")
                    ai_choice = ai_rerank_index(gname, scored)
                    if ai_choice:
                        cand = scored[ai_choice-1]
                        cand_score = cand[1]
                        cand_overlap = cand[4]
                        # Accept only if candidate meets thresholds
                        if cand_score >= 75 or (cand_score >= 60 and cand_overlap):
                            accepted = True
                            chosen_app = cand[3]
                            chosen_score = cand_score
                            print(f"🤖 AI accepted index {ai_choice} -> {cand[0]} (score {round(cand_score,2)})")
                        else:
                            print(f"🤖 AI selected index {ai_choice} but candidate did not meet acceptance thresholds (score {round(cand_score,2)}, overlap {cand_overlap})")
                    else:
                        print("🤖 AI returned None or couldn't decide")
            else:
                # best_score < 70: only call AI if google name contains non-generic keywords (so it's worth checking)
                gkeywords = extract_keywords(normalize(gname))
                if gkeywords:
                    print("⚠️ low fuzzy score (<70) but google has keywords -> invoking AI re-ranker")
                    ai_choice = ai_rerank_index(gname, scored)
                    if ai_choice:
                        cand = scored[ai_choice-1]
                        cand_score = cand[1]
                        cand_overlap = cand[4]
                        if cand_score >= 75 or (cand_score >= 60 and cand_overlap):
                            accepted = True
                            chosen_app = cand[3]
                            chosen_score = cand_score
                            print(f"🤖 AI accepted index {ai_choice} -> {cand[0]} (score {round(cand_score,2)})")
                        else:
                            print(f"🤖 AI selected index {ai_choice} but candidate fails acceptance thresholds (score {round(cand_score,2)}, overlap {cand_overlap})")
                    else:
                        print("🤖 AI returned None or couldn't decide")
                else:
                    print("❌ Low fuzzy score and no useful keywords — skipping AI (keep google-only)")

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
            print(f"✅ Final match: {chosen_app.get('title')} (Score {round(chosen_score,2)})")
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
            print("❌ Final: no confident match")

        merged_rows.append(merged)

        if (i+1) % 10 == 0:
            print(f"Progress: processed {i+1} apps")
        time.sleep(0.5)  # throttle

    df = pd.DataFrame(merged_rows)
    df.to_csv(out_csv, index=False)
    print("\n✅ Finished merging!")
    print(f"Saved to {out_csv} with {len(df)} rows")
    return df

# ------------------ Run ------------------
if __name__ == "__main__":
    df = merge_google_apple()
    print("\nSample merged rows:")
    cols = ["App","App_apple","Rating_google","Rating_apple","Match_Score","Source"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(10).to_string(index=False))
