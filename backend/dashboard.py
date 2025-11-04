import os
import sys
from datetime import datetime, timezone, timedelta

print("[Dashboard] Démarrage du dashboard...")
print(f"[Dashboard] Python version: {sys.version}")
print(f"[Dashboard] Working directory: {os.getcwd()}")

try:
    import pandas as pd

    print("[Dashboard] ✓ pandas importé")
except ImportError as e:
    print(f"[Dashboard] ✗ Erreur import pandas: {e}")
    sys.exit(1)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    print("[Dashboard] ✓ FastAPI importé")
except ImportError as e:
    print(f"[Dashboard] ✗ Erreur import FastAPI: {e}")
    sys.exit(1)

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    print("[Dashboard] ✓ Jinja2 importé")
except ImportError as e:
    print(f"[Dashboard] ✗ Erreur import Jinja2: {e}")
    sys.exit(1)

try:
    from sqlalchemy import create_engine, text

    print("[Dashboard] ✓ SQLAlchemy importé")
except ImportError as e:
    print(f"[Dashboard] ✗ Erreur import SQLAlchemy: {e}")
    sys.exit(1)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bot:botpass@db:5433/trading")
TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", "AAPL,TSLA,NVDA").split(",") if t.strip()]

print(f"[Dashboard] DATABASE_URL: {DATABASE_URL}")
print(f"[Dashboard] TICKERS: {TICKERS}")

max_retries = 5
retry_delay = 2
engine = None

for attempt in range(max_retries):
    try:
        print(f"[Dashboard] Tentative de connexion à la base de données ({attempt + 1}/{max_retries})...")
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
        # Test de connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[Dashboard] ✓ Connexion à la base de données réussie")
        break
    except Exception as e:
        print(f"[Dashboard] ✗ Erreur de connexion (tentative {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            import time

            print(f"[Dashboard] Nouvelle tentative dans {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            print("[Dashboard] ✗ Impossible de se connecter à la base de données après plusieurs tentatives")
            sys.exit(1)

app = FastAPI(title="Trading Dashboard")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("[Dashboard] ✓ Répertoire static monté")
else:
    print("[Dashboard] ⚠ Répertoire static non trouvé, ignoré")

if not os.path.exists("templates"):
    print("[Dashboard] ✗ Répertoire templates non trouvé!")
    sys.exit(1)

templates_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"])
)
print("[Dashboard] ✓ Templates chargés")


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
        last_eq = con.execute(
            text("select timestamp as ts, equity from portfolio_history order by timestamp desc limit 1")).fetchone()
        # positions
        pos_rows = con.execute(text("select symbol, qty, entry from positions order by symbol")).fetchall()
        # trades récents
        trades = con.execute(text("""
                                  select timestamp as ts, symbol, side, quantity as qty, price, fees as fee
                                  from trades
                                  order by timestamp desc limit 50
                                  """)).fetchall()
        try:
            # derniers indicateurs par ticker
            inds = con.execute(text("""
                                    select distinct
                                    on (symbol) symbol, timestamp as ts, close, sma_s, sma_l, rsi
                                    from candles_1m
                                    order by symbol, timestamp desc
                                    """)).fetchall()
        except Exception as e:
            print(f"[Dashboard] ⚠ Erreur lors de la récupération des indicateurs: {e}")
            inds = []

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
            "safety_score": safety_score
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
            text("select timestamp as ts, equity from portfolio_history where timestamp >= :since order by timestamp"),
            con, params={"since": since}
        )
    return JSONResponse({
        "ts": [ts.isoformat() for ts in df["ts"]] if not df.empty else [],
        "equity": df["equity"].tolist() if not df.empty else []
    })


@app.get("/api/indicators.json")
def indicators_json(symbol: str):
    try:
        with engine.begin() as con:
            df = pd.read_sql_query(
                text("""
                     select timestamp as ts, close, sma_s, sma_l, rsi
                     from candles_1m
                     where symbol = :s
                     order by timestamp desc
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
    except Exception as e:
        print(f"[Dashboard] ⚠ Erreur lors de la récupération des indicateurs pour {symbol}: {e}")
        return JSONResponse({
            "ts": [],
            "close": [],
            "sma_s": [],
            "sma_l": [],
            "rsi": [],
        })


@app.get("/api/positions.json")
def positions_json():
    with engine.begin() as con:
        pos_rows = con.execute(text("select symbol, qty, entry from positions order by symbol")).fetchall()
        try:
            inds = con.execute(text("""
                                    select distinct
                                    on (symbol) symbol, timestamp as ts, close, sma_s, sma_l, rsi
                                    from candles_1m
                                    order by symbol, timestamp desc
                                    """)).fetchall()
        except Exception as e:
            print(f"[Dashboard] ⚠ Erreur lors de la récupération des indicateurs: {e}")
            inds = []

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
            "safety_score": safety_score
        })

    return JSONResponse({"positions": positions})


@app.get("/api/kpis.json")
def kpis_json():
    with engine.begin() as con:
        last_eq = con.execute(
            text("select timestamp as ts, equity from portfolio_history order by timestamp desc limit 1")).fetchone()
        pos_count = con.execute(text("select count(*) from positions")).fetchone()[0]

    return JSONResponse({
        "equity": float(last_eq[1]) if last_eq else 0.0,
        "equity_ts": last_eq[0].strftime("%Y-%m-%d %H:%M:%S") if last_eq else "—",
        "positions_count": pos_count
    })


@app.get("/health")
def health_check():
    """Endpoint de healthcheck pour vérifier que le dashboard fonctionne"""
    try:
        # Test de connexion à la base de données
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return JSONResponse({
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, status_code=503)


print("[Dashboard] ✓ Dashboard prêt à démarrer")
print("[Dashboard] Endpoints disponibles:")
print("[Dashboard]   - GET  /")
print("[Dashboard]   - GET  /health")
print("[Dashboard]   - GET  /api/equity.json")
print("[Dashboard]   - GET  /api/indicators.json")
print("[Dashboard]   - GET  /api/positions.json")
print("[Dashboard]   - GET  /api/kpis.json")

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8080"))

    print(f"[Dashboard] Démarrage du serveur sur {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
