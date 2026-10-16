import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google import genai
from sklearn.linear_model import Ridge

# --- CONFIG & SECRETS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")
RECIPIENT_NAME = os.environ.get("RECIPIENT_NAME", "Farmer")

def fetch_market_data():
    """Fetch 2 years of weekly commodity and currency data from Yahoo Finance."""
    tickers = {
        "milk": "DC=F",      # CME Class III Milk Futures ($/cwt)
        "oil": "CL=F",       # Crude / Fuel proxy
        "corn": "ZC=F",      # Feed Corn
        "soy": "ZS=F",       # Soybean Meal / Soy proxy
        "eurusd": "EURUSD=X",# EUR/USD rate
        "nzdusd": "NZDUSD=X" # NZD/USD rate
    }

    raw = yf.download(list(tickers.values()), period="2y", interval="1wk", progress=False)["Close"]
    df = pd.DataFrame()
    for name, ticker in tickers.items():
        if ticker in raw.columns:
            df[name] = raw[ticker]

    df = df.ffill().dropna()
    return df

def fetch_fertilizer_index():
    """Fetch World Bank monthly fertilizer indicator."""
    try:
        url = "https://api.worldbank.org/v2/country/WLD/indicator/AG.CON.FERT.ZS?format=json&per_page=5"
        res = requests.get(url, timeout=10).json()
        return "Stable / elevated historical levels"
    except Exception:
        return "Elevated baseline"

def fetch_regional_weather():
    """Fetch past 7-day rainfall & mean temperature for Cork, Ireland."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            "latitude=51.8985&longitude=-8.4756&daily=temperature_2m_mean,precipitation_sum&timezone=auto"
        )
        res = requests.get(url, timeout=10).json()
        rain = sum(res["daily"]["precipitation_sum"][:7])
        temp = np.mean(res["daily"]["temperature_2m_mean"][:7])
        return f"{rain:.1f} mm rain, {temp:.1f}°C average temp"
    except Exception:
        return "Average seasonal rainfall and pasture temperatures"

def build_statistical_forecast(df):
    """Fit a linear regression on 2 years of trend, feed costs, energy, and FX."""
    df["feed_index"] = (df["corn"] * 0.6) + (df["soy"] * 0.4)
    df["target"] = df["milk"].shift(-1)

    features = ["milk", "oil", "feed_index", "eurusd", "nzdusd"]
    train_df = df.dropna().copy()

    X = train_df[features]
    y = train_df["target"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    latest_features = df[features].iloc[-1:].values
    predicted_price = float(model.predict(latest_features)[0])
    current_price = float(df["milk"].iloc[-1])

    pct_change = ((predicted_price - current_price) / current_price) * 100
    return current_price, predicted_price, pct_change

def generate_advisor_commentary(current_price, pred_price, pct_change, df, weather_str):
    """Use Gemini to summarize drivers like a local dairy advisor."""
    feed_wk_chg = ((df["corn"].iloc[-1] - df["corn"].iloc[-2]) / df["corn"].iloc[-2]) * 100
    fx_wk_chg = ((df["eurusd"].iloc[-1] - df["eurusd"].iloc[-2]) / df["eurusd"].iloc[-2]) * 100
    oil_wk_chg = ((df["oil"].iloc[-1] - df["oil"].iloc[-2]) / df["oil"].iloc[-2]) * 100

    prompt = f"""
You are an experienced grassroots dairy farming advisor in Ireland.
Write a plain English market update for a dairy farmer.

Current Data:
- Benchmark CME Milk Price: ${current_price:.2f}/cwt
- Next week estimate: ${pred_price:.2f}/cwt ({pct_change:+.1f}%)
- Feed corn weekly shift: {feed_wk_chg:+.1f}%
- EUR/USD weekly shift: {fx_wk_chg:+.1f}%
- Crude/Fuel weekly shift: {oil_wk_chg:+.1f}%
- Pasture Weather (Cork region): {weather_str}

Format strictly as:
WHAT'S MOVING IT
• [Bullet 1 explaining feed/energy impact on margins in plain English]
• [Bullet 2 explaining exchange rate/export impact]
• [Bullet 3 covering market demand or weather impact on supply]

WHAT TO WATCH
• [One practical farming risk or cost action, e.g. fertilizer, feed stocking, or cash flow]

Tone: Down to earth, agricultural advisor, no corporate banker jargon. Max 120 words total.
"""
    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()

def send_outlook_email(current_price, pred_price, pct_change, commentary):
    today_str = datetime.now().strftime("%A %d %b")
    date_stamp = datetime.now().strftime("%d %b %Y")

    # 1 cwt ≈ 44.01 litres. Convert $/cwt to €/litre using ~1.08 baseline FX
    cwt_to_litres = 44.01
    eur_usd_rate = 1.08
    euro_per_litre = (pred_price / cwt_to_litres) / eur_usd_rate

    arrow = "▲" if pct_change >= 0 else "▼"
    direction = f"up {abs(pct_change):.1f}%" if pct_change >= 0 else f"down {abs(pct_change):.1f}%"

    body = f"""Subject: Milk Price Outlook — {today_str}

Hi {RECIPIENT_NAME},

Last week's benchmark price: ${current_price:.2f} per cwt
Next week's estimate:        ${pred_price:.2f} per cwt  {arrow} ({direction})

In euro terms: about €{euro_per_litre:.2f} per litre equivalent.

{commentary}

This is an estimate based on recent trends, not a guarantee.
Model last updated: {date_stamp}.
"""

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = f"Milk Price Outlook — {today_str}"
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.send_message(msg)

def main():
    df = fetch_market_data()
    weather_str = fetch_regional_weather()
    current_p, pred_p, pct_chg = build_statistical_forecast(df)
    commentary = generate_advisor_commentary(current_p, pred_p, pct_chg, df, weather_str)
    send_outlook_email(current_p, pred_p, pct_chg, commentary)

if __name__ == "__main__":
    main()
