import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bot:botpass@db:5432/trading")
TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", "AAPL,TSLA,NVDA").split(",") if t.strip()]

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="Trading Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"])
)


def fmt2(x, nd=2):
    try:
        if x is None: return "—"
        return f"{float(x):.{nd}f}"
    except:
        return "—"

def calculate_safety_score(indicators: dict, price: float, entry: float, highest: float) -> float:
    """
    Calcule un score de sûreté 0-100 pour une position en profit.
    Plus le score est élevé, plus il est sûr de garder la position.
    Score < 40 : Risque de baisse, vendre
    Score >= 40 : Potentiel de hausse, garder
    """
    score = 0.0
    
    # Facteur 1: MACD momentum positif (20 points)
    macd_hist = indicators.get("macd_hist")
    macd_val = indicators.get("macd")
    macd_sig = indicators.get("macd_signal")
    if macd_hist and macd_val and macd_sig:
        if macd_hist > 0 and macd_val > macd_sig:
            score += 20
        elif macd_hist > 0:
            score += 10
    
    # Facteur 2: RSI dans zone de force saine (20 points)
    rsi = indicators.get("rsi")
    if rsi:
        if 50 <= rsi <= 70:
            score += 20
        elif 45 <= rsi <= 75:
            score += 10
        elif rsi > 75:
            score += 0
    
    # Facteur 3: Tendance haussière (15 points)
    sma_s = indicators.get("sma_s")
    sma_l = indicators.get("sma_l")
    if sma_s and sma_l:
        if price > sma_s and price > sma_l:
            score += 15
        elif price > sma_s:
            score += 8
    
    # Facteur 4: Stochastic momentum positif (15 points) - simulé si pas disponible
    score += 10  # Valeur par défaut
    
    # Facteur 5: Volatilité stable (15 points) - simulé
    score += 10  # Valeur par défaut
    
    # Facteur 6: Distance du plus haut (15 points)
    if highest > entry:
        distance_from_high = ((highest - price) / highest) * 100
        if distance_from_high < 0.5:
            score += 15
        elif distance_from_high < 1.0:
            score += 10
        elif distance_from_high < 2.0:
            score += 5
    
    return min(100, max(0, score))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    with engine.begin() as con:
        # equity dernière valeur
        last_eq = con.execute(text("select ts, equity from equity order by ts desc limit 1")).fetchone()
        # positions
        pos_rows = con.execute(text("select symbol, qty, entry from positions order by symbol")).fetchall()
        # trades récents
        trades = con.execute(text("""
                                  select ts, symbol, side, qty, price, fee
                                  from trades
                                  order by ts desc limit 50
                                  """)).fetchall()
        # derniers indicateurs par ticker
        inds = con.execute(text("""
                                select distinct
                                on (symbol) symbol, ts, close, sma_s, sma_l, rsi
                                from candles_1m
                                order by symbol, ts desc
                                """)).fetchall()

    positions = []
    for sym, qty, entry in pos_rows:
        # prix courant si dispo via dernière candle
        found = next((r for r in inds if r[0] == sym), None)
        px = float(found[2]) if found else None
        pnl = (px - float(entry)) * float(qty) if (px and qty) else 0.0
        
        safety_score = None
        if px and pnl > 0:
            # Récupérer les indicateurs pour ce symbole
            indicators = {
                "rsi": float(found[5]) if found and found[5] else None,
                "sma_s": float(found[3]) if found and found[3] else None,
                "sma_l": float(found[4]) if found and found[4] else None,
                "macd": None,  # Pas disponible dans candles_1m
                "macd_signal": None,
                "macd_hist": None
            }
            # Utiliser le prix actuel comme highest (approximation)
            highest = px
            safety_score = calculate_safety_score(indicators, px, float(entry), highest)
        
        positions.append({
            "symbol": sym,
            "qty": qty,
            "entry": entry,
            "last": px,
            "pnl": pnl,
            "safety_score": safety_score  # Ajout du safety score
        })

    template = templates_env.get_template("index.html")
    html = template.render(
        tickers=TICKERS,
        equity_ts=last_eq[0].strftime("%Y-%m-%d %H:%M:%S") if last_eq else "—",
        equity_val=fmt2(last_eq[1]) if last_eq else "—",
        positions=positions,
        trades=trades,
        indicators=[{
            "symbol": r[0],
            "ts": r[1].strftime("%Y-%m-%d %H:%M"),
            "close": fmt2(r[2]),
            "sma_s": fmt2(r[3]),
            "sma_l": fmt2(r[4]),
            "rsi": fmt2(r[5], 1),
        } for r in inds]
    )
    return HTMLResponse(html)


@app.get("/api/equity.json")
def equity_json(hours: int = 24):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    with engine.begin() as con:
        df = pd.read_sql_query(
            text("select ts, equity from equity where ts >= :since order by ts"),
            con, params={"since": since}
        )
    return JSONResponse({
        "ts": [ts.isoformat() for ts in df["ts"]] if not df.empty else [],
        "equity": df["equity"].tolist() if not df.empty else []
    })


@app.get("/api/indicators.json")
def indicators_json(symbol: str):
    with engine.begin() as con:
        df = pd.read_sql_query(
            text("""
                 select ts, close, sma_s, sma_l, rsi
                 from candles_1m
                 where symbol = :s
                 order by ts desc
                     limit 600
                 """), con, params={"s": symbol}
        )
    df = df.sort_values("ts")
    return JSONResponse({
        "ts": [pd.to_datetime(t).isoformat() for t in df["ts"]],
        "close": [float(x) for x in df["close"]],
        "sma_s": [None if pd.isna(x) else float(x) for x in df["sma_s"]],
        "sma_l": [None if pd.isna(x) else float(x) for x in df["sma_l"]],
        "rsi": [None if pd.isna(x) else float(x) for x in df["rsi"]],
    })


@app.get("/api/positions.json")
def positions_json():
    with engine.begin() as con:
        pos_rows = con.execute(text("select symbol, qty, entry from positions order by symbol")).fetchall()
        inds = con.execute(text("""
                                select distinct
                                on (symbol) symbol, ts, close, sma_s, sma_l, rsi
                                from candles_1m
                                order by symbol, ts desc
                                """)).fetchall()

    positions = []
    for sym, qty, entry in pos_rows:
        found = next((r for r in inds if r[0] == sym), None)
        px = float(found[2]) if found else None
        pnl = (px - float(entry)) * float(qty) if (px and qty) else 0.0
        
        safety_score = None
        if px and pnl > 0:
            indicators = {
                "rsi": float(found[5]) if found and found[5] else None,
                "sma_s": float(found[3]) if found and found[3] else None,
                "sma_l": float(found[4]) if found and found[4] else None,
                "macd": None,
                "macd_signal": None,
                "macd_hist": None
            }
            highest = px
            safety_score = calculate_safety_score(indicators, px, float(entry), highest)
        
        positions.append({
            "symbol": sym,
            "qty": float(qty),
            "entry": float(entry),
            "last": px,
            "pnl": pnl,
            "safety_score": safety_score  # Ajout du safety score
        })

    return JSONResponse({"positions": positions})


@app.get("/api/kpis.json")
def kpis_json():
    with engine.begin() as con:
        last_eq = con.execute(text("select ts, equity from equity order by ts desc limit 1")).fetchone()
        pos_count = con.execute(text("select count(*) from positions")).fetchone()[0]

    return JSONResponse({
        "equity": float(last_eq[1]) if last_eq else 0.0,
        "equity_ts": last_eq[0].strftime("%Y-%m-%d %H:%M:%S") if last_eq else "—",
        "positions_count": pos_count
    })
