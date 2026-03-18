import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import json
import yaml


load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SENTIMENT_FILE  = "data/gold/company_sentiment_scores.csv"
SIGNALS_FILE    = "data/gold/investment_signals.csv"
FINANCIALS_FILE = "data/silver/processed_financials.csv"
OUTPUT_DIR      = "data/gold/reports"

GROQ_MODEL = "llama-3.1-8b-instant"




# ── Prompt Version ────────────────────────────────────────────────────────────

PROMPT_VERSION = {
    "version":         "v1.0",
    "task":            "weekly_sector_report",
    "model_name":      GROQ_MODEL,
    "created_at":      "2026-03-18",
    "is_active":       True,
    "parameters": {
        "max_tokens":  500,
        "temperature": 0.3,
    },
    "prompt_template": """You are a senior financial analyst covering Indian automobile stocks.

Today is {report_date}. Below is the latest sentiment and financial data for 5 major Indian auto companies:

{company_block}

Sector average sentiment: {sector_avg}/10
Top sentiment company: {top_company} ({top_score}/10)
Weakest sentiment company: {bottom_company} ({bottom_score}/10)

Write a professional 3-paragraph sector intelligence report covering:
1. Overall sector sentiment and what the data suggests about market conditions
2. Standout companies — both positive and negative — and what's driving the divergence
3. Investment outlook based on the signals, RSI levels, and profitability data

Rules:
- Be specific, cite the numbers from the data
- Tone must be objective and data-driven
- No disclaimers, no generic filler
- Maximum 250 words"""
}

PROMPT_VERSIONS_DIR = "data/bronze/prompt_versions"


# ── Helpers ───────────────────────────────────────────────────────────────────

def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

def format_date():
    return datetime.now().strftime("%d %B %Y")



def save_prompt_version(version: dict):
    os.makedirs(PROMPT_VERSIONS_DIR, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(
        PROMPT_VERSIONS_DIR,
        f"prompt_{version['task']}_{version['version']}.json"
    )
    with open(json_path, "w") as f:
        json.dump(version, f, indent=2)

    # Save as YAML
    yaml_path = os.path.join(
        PROMPT_VERSIONS_DIR,
        f"prompt_{version['task']}_{version['version']}.yaml"
    )
    with open(yaml_path, "w") as f:
        yaml.dump(version, f, default_flow_style=False, allow_unicode=True)

    print(f"  Prompt version saved → {json_path}")
    print(f"  Prompt version saved → {yaml_path}")

# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data() -> dict:
    sentiment_df  = pd.read_csv(SENTIMENT_FILE)
    signals_df    = pd.read_csv(SIGNALS_FILE)
    financials_df = pd.read_csv(FINANCIALS_FILE)

    merged = signals_df.merge(
        financials_df[[
            "company", "profit_margin_pct", "revenue_growth_pct"
        ]],
        on="company", how="left"
    )

    sector_avg    = round(sentiment_df["sentiment_score"].mean(), 2)
    top_company   = sentiment_df.loc[sentiment_df["sentiment_score"].idxmax(), "company"]
    top_score     = round(sentiment_df["sentiment_score"].max(), 2)
    bottom_company = sentiment_df.loc[sentiment_df["sentiment_score"].idxmin(), "company"]
    bottom_score  = round(sentiment_df["sentiment_score"].min(), 2)

    return {
        "merged":          merged,
        "sector_avg":      sector_avg,
        "top_company":     top_company,
        "top_score":       top_score,
        "bottom_company":  bottom_company,
        "bottom_score":    bottom_score,
        "report_date":     format_date(),
        "generated_at":    now(),
    }

# ── Build Prompt ──────────────────────────────────────────────────────────────

def build_prompt(data: dict) -> str:
    merged = data["merged"]

    company_lines = []
    for _, row in merged.iterrows():
        margin = row.get("profit_margin_pct")
        growth = row.get("revenue_growth_pct")
        company_lines.append(
            f"- {row['company']}: sentiment={row['sentiment_score']:.2f}/10, "
            f"signal={row['signal']}, RSI={row['rsi_14']:.1f}, "
            f"price_vs_MA20={row['price_vs_ma20']*100:.1f}%, "
            f"profit_margin={f'{margin:.1f}%' if pd.notna(margin) else 'N/A'}, "
            f"revenue_growth={f'{growth:.1f}%' if pd.notna(growth) else 'N/A'}, "
            f"profitability_score={row['profitability_score']}/10"
        )

    company_block = "\n".join(company_lines)

    prompt = f"""You are a senior financial analyst covering Indian automobile stocks.

Today is {data['report_date']}. Below is the latest sentiment and financial data for 5 major Indian auto companies:

{company_block}

Sector average sentiment: {data['sector_avg']}/10
Top sentiment company: {data['top_company']} ({data['top_score']}/10)
Weakest sentiment company: {data['bottom_company']} ({data['bottom_score']}/10)

Write a professional 3-paragraph sector intelligence report covering:
1. Overall sector sentiment and what the data suggests about market conditions
2. Standout companies — both positive and negative — and what's driving the divergence
3. Investment outlook based on the signals, RSI levels, and profitability data

Rules:
- Be specific, cite the numbers from the data
- Tone must be objective and data-driven
- No disclaimers, no generic filler
- Maximum 250 words
"""
    return prompt

# ── Call Groq ─────────────────────────────────────────────────────────────────

def call_groq(prompt: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print(f"  Calling Groq ({GROQ_MODEL})...")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ── Save ──────────────────────────────────────────────────────────────────────

def save(report_text: str, data: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    # TXT
    txt_path = os.path.join(OUTPUT_DIR, f"report_{date_str}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # JSON — for Django API
    json_path = os.path.join(OUTPUT_DIR, f"report_{date_str}.json")
    json_data = {
        "report_date":          data["report_date"],
        "generated_at":         data["generated_at"],
        "sector_avg_sentiment": data["sector_avg"],
        "top_company":          data["top_company"],
        "top_score":            data["top_score"],
        "bottom_company":       data["bottom_company"],
        "bottom_score":         data["bottom_score"],
        "report_text":          report_text,
        "generated_by":         GROQ_MODEL,
        "companies":            data["merged"][[
            "company", "signal", "sentiment_score",
            "momentum", "rsi_14", "price_vs_ma20",
            "profitability_score", "reasoning"
        ]].to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(f"  Saved → {txt_path}")
    print(f"  Saved → {json_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"Gold Report Generation (Groq / {GROQ_MODEL})")
    print(f"{'='*50}\n")

    print("  Loading data...")
    save_prompt_version(PROMPT_VERSION)
    print(f"  Using prompt version: {PROMPT_VERSION['version']}")
    data = load_data()

    prompt = build_prompt(data)
    print(f"  Prompt built ({len(prompt)} chars)")

    report_text = call_groq(prompt)

    print(f"\n{'─'*50}")
    print(report_text)
    print(f"{'─'*50}\n")

    save(report_text, data)
    print("\nReport generation complete.\n")

if __name__ == "__main__":
    run()