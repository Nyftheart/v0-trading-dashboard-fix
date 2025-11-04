#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot de trading avancé avec protection contre les pertes et optimisation du rate limiting.
Nouvelles fonctionnalités :
- Stop-loss et take-profit automatiques
- Trailing stop pour sécuriser les gains
- Position sizing dynamique basé sur ATR (volatilité)
- Score de confiance multi-facteurs
- MACD, Bollinger Bands, ATR
- Protection max drawdown
- Optimisation 60 req/min
"""
import os, time, math, requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, deque

import pytz
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from multi_timeframe import integrate_mtf_with_bot

FINNHUB_QUOTE = "https://finnhub.io/api/v1/quote"

# ========= Utils =========
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def ts_str(dt: Optional[datetime] = None) -> str:
    return (dt or utcnow()).strftime("%Y-%m-%d %H:%M:%S")

def fmt(x, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return "—"
        return f"{xf:.{nd}f}"
    except:
        return "—"

def parse_active_hours(s: str) -> Tuple[float, float]:
    """Parse "9-17.5" -> (9.0, 17.5)."""
    try:
        part = s.strip().replace(" ", "")
        a, b = part.split("-")
        return float(a), float(b)
    except Exception:
        return 9.0, 17.5

def within_paris_session(active_hours: Tuple[float, float]) -> tuple[bool, str]:
    """Retourne (is_open, reason). reason explique pourquoi c'est fermé (week-end/hours)."""
    tz_paris = pytz.timezone("Europe/Paris")
    now_paris = datetime.now(tz_paris)
    weekday = now_paris.weekday()
    hour_local = now_paris.hour + now_paris.minute / 60.0
    start_h, end_h = active_hours
    if weekday >= 5:
        return False, f"week-end (jour {weekday})"
    if not (start_h <= hour_local <= end_h):
        return False, f"horaire fermé ({hour_local:.2f}h)"
    return True, ""

def fetch_quote(sym: str, key: str) -> dict:
    r = requests.get(FINNHUB_QUOTE, params={"symbol": sym, "token": key}, timeout=10)
    r.raise_for_status()
    return r.json()

def key_fp(k: str) -> str:
    """Empreinte lisible d'une clé (8 derniers chars)."""
    return k[-8:] if k else "????????"

# ========= Indicateurs avancés =========
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    """RSI de Wilder (EWMA) robuste."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False, min_periods=1).mean()
    roll_down = roll_down.replace(0, pd.NA)
    rs = roll_up / roll_down
    rsi_val = 100 - (100 / (1 + rs))
    rsi_val = rsi_val.fillna(100).fillna(50)
    return rsi_val

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range - mesure de volatilité"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False, min_periods=length).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, length: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    middle = sma(series, length)
    std = series.rolling(length, min_periods=length).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(length, min_periods=length).min()
    highest_high = high.rolling(length, min_periods=length).max()
    stoch = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return stoch.fillna(50)

# ========= DB helpers =========
def upsert_candle(engine, symbol: str, ts, close, sma_s, sma_l, rsi_v):
    with engine.begin() as con:
        con.execute(text("""
            insert into candles_1m(symbol, ts, close, sma_s, sma_l, rsi)
            values (:sym, :ts, :c, :sma_s, :sma_l, :rsi)
            on conflict (symbol, ts) do update
            set close=excluded.close, sma_s=excluded.sma_s, sma_l=excluded.sma_l, rsi=excluded.rsi
        """), {"sym": symbol, "ts": ts, "c": close, "sma_s": sma_s, "sma_l": sma_l, "rsi": rsi_v})

def insert_tick(engine, symbol: str, ts, price: float):
    with engine.begin() as con:
        con.execute(text("""
            insert into ticks(symbol, ts, price)
            values (:s, :ts, :p)
            on conflict (symbol, ts) do nothing
        """), {"s": symbol, "ts": ts, "p": price})

def get_position(engine, symbol: str):
    with engine.begin() as con:
        return con.execute(text("select qty, entry from positions where symbol=:s"), {"s": symbol}).fetchone()

def insert_trade(engine, ts, symbol: str, side: str, qty: float, price: float, fee: float):
    with engine.begin() as con:
        con.execute(text("""
            insert into trades(ts, symbol, side, qty, price, fee)
            values(:ts,:s,:side,:q,:p,:f)
        """), {"ts": ts, "s": symbol, "side": side, "q": qty, "p": price, "f": fee})

def upsert_position_buy(engine, symbol: str, qty: float, entry: float):
    with engine.begin() as con:
        con.execute(text("insert into positions(symbol, qty, entry) values(:s,:q,:e)"),
                    {"s": symbol, "q": qty, "e": entry})

def delete_position(engine, symbol: str):
    with engine.begin() as con:
        con.execute(text("delete from positions where symbol=:s"), {"s": symbol})

def insert_equity(engine, ts, equity: float):
    with engine.begin() as con:
        con.execute(text("""
            insert into equity(ts, equity) values(:ts,:eq)
            on conflict (ts) do nothing
        """), {"ts": ts, "eq": equity})

# ========= Indicateurs avancés depuis DB =========
def compute_advanced_indicators(engine, symbol: str, fast_mode: bool) -> Optional[dict]:
    """
    Calcule tous les indicateurs avancés :
    - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic
    - Volatilité, drawdown, score de confiance
    - Multi-timeframe analysis
    """
    with engine.begin() as con:
        df = pd.read_sql_query(
            text("select ts, price from ticks where symbol=:s and ts >= now() - interval '3 days' order by ts"),
            con, params={"s": symbol}, parse_dates=["ts"]
        )

    if df.empty or len(df) < 50:
        return None

    df = df.set_index("ts").sort_index()
    close_1m = df["price"].resample("1min").last().ffill()
    df1 = pd.DataFrame({"close": close_1m})
    
    # Approximation high/low depuis close (pour ATR et Stochastic)
    df1["high"] = df1["close"].rolling(3, min_periods=1).max()
    df1["low"] = df1["close"].rolling(3, min_periods=1).min()

    if fast_mode:
        s_s, s_l, r_l = 5, 20, 7
        macd_fast, macd_slow, macd_sig = 6, 13, 5
        bb_len = 10
        atr_len = 7
        stoch_len = 7
    else:
        s_s, s_l, r_l = 10, 50, 14
        macd_fast, macd_slow, macd_sig = 12, 26, 9
        bb_len = 20
        atr_len = 14
        stoch_len = 14

    # Indicateurs de base
    df1["sma_s"] = sma(df1["close"], s_s)
    df1["sma_l"] = sma(df1["close"], s_l)
    df1["ema_s"] = ema(df1["close"], s_s)
    df1["ema_l"] = ema(df1["close"], s_l)
    df1["rsi"] = rsi_wilder(df1["close"], r_l).bfill()
    
    # MACD
    macd_line, signal_line, histogram = macd(df1["close"], macd_fast, macd_slow, macd_sig)
    df1["macd"] = macd_line
    df1["macd_signal"] = signal_line
    df1["macd_hist"] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(df1["close"], bb_len)
    df1["bb_upper"] = bb_upper
    df1["bb_middle"] = bb_middle
    df1["bb_lower"] = bb_lower
    
    # ATR (volatilité)
    df1["atr"] = atr(df1["high"], df1["low"], df1["close"], atr_len)
    
    # Stochastic
    df1["stoch"] = stochastic(df1["high"], df1["low"], df1["close"], stoch_len)
    
    # Drawdown et recent high
    lookback = 60 if not fast_mode else 30
    df1["recent_high"] = df1["close"].rolling(lookback, min_periods=1).max()
    df1["drawdown_pct"] = ((df1["recent_high"] - df1["close"]) / df1["recent_high"] * 100).clip(lower=0)

    row = df1.iloc[-1]
    
    # Score de confiance (0-100)
    confidence_score = calculate_confidence_score(row, df1)
    
    mtf_score = 50.0  # Valeur par défaut
    mtf_results = {}
    try:
        # Préparer les données pour MTF
        df_for_mtf = df.copy()
        df_for_mtf = df_for_mtf.set_index("ts")
        df_for_mtf = df_for_mtf.resample("1min").agg({"price": ["first", "max", "min", "last"]})
        df_for_mtf.columns = ["open", "high", "low", "close"]
        df_for_mtf = df_for_mtf.dropna()
        
        if len(df_for_mtf) >= 100:  # Besoin d'assez de données
            mtf_score, mtf_results = integrate_mtf_with_bot(df_for_mtf, fast_mode)
    except Exception as e:
        print(f"[{ts_str()}] MTF Analysis error: {e}", flush=True)
    
    out = {
        "ts": df1.index[-1].to_pydatetime(),
        "close": float(row["close"]),
        "sma_s": None if pd.isna(row["sma_s"]) else float(row["sma_s"]),
        "sma_l": None if pd.isna(row["sma_l"]) else float(row["sma_l"]),
        "ema_s": None if pd.isna(row["ema_s"]) else float(row["ema_s"]),
        "ema_l": None if pd.isna(row["ema_l"]) else float(row["ema_l"]),
        "rsi": None if pd.isna(row["rsi"]) else float(row["rsi"]),
        "macd": None if pd.isna(row["macd"]) else float(row["macd"]),
        "macd_signal": None if pd.isna(row["macd_signal"]) else float(row["macd_signal"]),
        "macd_hist": None if pd.isna(row["macd_hist"]) else float(row["macd_hist"]),
        "bb_upper": None if pd.isna(row["bb_upper"]) else float(row["bb_upper"]),
        "bb_middle": None if pd.isna(row["bb_middle"]) else float(row["bb_middle"]),
        "bb_lower": None if pd.isna(row["bb_lower"]) else float(row["bb_lower"]),
        "atr": None if pd.isna(row["atr"]) else float(row["atr"]),
        "stoch": None if pd.isna(row["stoch"]) else float(row["stoch"]),
        "recent_high": float(row["recent_high"]),
        "drawdown_pct": float(row["drawdown_pct"]),
        "confidence_score": confidence_score,
        "volatility_regime": classify_volatility(row["atr"], df1["atr"]) if not pd.isna(row["atr"]) else "unknown",
        "mtf_score": mtf_score,
        "mtf_results": mtf_results
    }

    upsert_candle(engine, symbol, out["ts"], out["close"], out["sma_s"], out["sma_l"], out["rsi"])
    return out

def calculate_confidence_score(row: pd.Series, df: pd.DataFrame) -> float:
    """
    Calcule un score de confiance 0-100 basé sur plusieurs facteurs.
    Plus le score est élevé, plus le signal est fiable.
    """
    score = 0.0
    factors = 0
    
    # Facteur 1: Alignement des moyennes mobiles (20 points)
    if not pd.isna(row["sma_s"]) and not pd.isna(row["sma_l"]) and not pd.isna(row["ema_s"]) and not pd.isna(row["ema_l"]):
        if row["sma_s"] > row["sma_l"] and row["ema_s"] > row["ema_l"]:
            score += 20
        factors += 1
    
    # Facteur 2: RSI dans zone favorable (15 points)
    if not pd.isna(row["rsi"]):
        if 30 <= row["rsi"] <= 70:
            score += 15
        elif 40 <= row["rsi"] <= 60:
            score += 10
        factors += 1
    
    # Facteur 3: MACD momentum (20 points)
    if not pd.isna(row["macd"]) and not pd.isna(row["macd_signal"]) and not pd.isna(row["macd_hist"]):
        if row["macd"] > row["macd_signal"] and row["macd_hist"] > 0:
            score += 20
        elif row["macd"] > row["macd_signal"]:
            score += 10
        factors += 1
    
    # Facteur 4: Position dans Bollinger Bands (15 points)
    if not pd.isna(row["bb_upper"]) and not pd.isna(row["bb_lower"]) and not pd.isna(row["bb_middle"]):
        bb_range = row["bb_upper"] - row["bb_lower"]
        if bb_range > 0:
            position = (row["close"] - row["bb_lower"]) / bb_range
            if 0.3 <= position <= 0.7:  # Zone médiane
                score += 15
            elif 0.2 <= position <= 0.8:
                score += 10
        factors += 1
    
    # Facteur 5: Stochastic (15 points)
    if not pd.isna(row["stoch"]):
        if 20 <= row["stoch"] <= 80:
            score += 15
        elif 30 <= row["stoch"] <= 70:
            score += 10
        factors += 1
    
    # Facteur 6: Volatilité stable (15 points)
    if not pd.isna(row["atr"]) and len(df) > 20:
        atr_mean = df["atr"].tail(20).mean()
        if not pd.isna(atr_mean) and atr_mean > 0:
            atr_ratio = row["atr"] / atr_mean
            if 0.7 <= atr_ratio <= 1.3:  # Volatilité stable
                score += 15
            elif 0.5 <= atr_ratio <= 1.5:
                score += 10
        factors += 1
    
    # Normaliser le score
    if factors > 0:
        return min(100, (score / factors) * (100 / 20))  # Normaliser à 100
    return 50.0

def classify_volatility(current_atr: float, atr_series: pd.Series) -> str:
    """Classifie le régime de volatilité: low, normal, high"""
    if pd.isna(current_atr) or len(atr_series) < 20:
        return "unknown"
    
    atr_mean = atr_series.tail(50).mean()
    atr_std = atr_series.tail(50).std()
    
    if pd.isna(atr_mean) or pd.isna(atr_std) or atr_std == 0:
        return "normal"
    
    z_score = (current_atr - atr_mean) / atr_std
    
    if z_score > 1.5:
        return "high"
    elif z_score < -0.5:
        return "low"
    else:
        return "normal"

# ========= Gestion des positions avancée =========
def calculate_risk_score(indicators: dict, current_equity: float, starting_equity: float) -> float:
    """
    Calcule un score de risque 0-100 pour déterminer la taille de position.
    0 = risque très faible (investir plus)
    100 = risque très élevé (investir moins)
    """
    risk_score = 0.0
    
    # Facteur 1: Volatilité (0-25 points de risque)
    volatility = indicators.get("volatility_regime", "normal")
    if volatility == "high":
        risk_score += 25
    elif volatility == "normal":
        risk_score += 12
    else:  # low
        risk_score += 5
    
    # Facteur 2: Score de confiance inversé (0-30 points de risque)
    confidence = indicators.get("confidence_score", 50)
    # Confiance faible = risque élevé
    risk_score += (100 - confidence) * 0.3
    
    # Facteur 3: RSI extrême (0-15 points de risque)
    rsi = indicators.get("rsi")
    if rsi:
        if rsi < 20 or rsi > 80:
            risk_score += 15
        elif rsi < 30 or rsi > 70:
            risk_score += 8
    
    # Facteur 4: MACD divergence (0-15 points de risque)
    macd_hist = indicators.get("macd_hist")
    if macd_hist:
        if abs(macd_hist) > 1.0:
            risk_score += 15
        elif abs(macd_hist) > 0.5:
            risk_score += 8
    
    # Facteur 5: Drawdown actuel du portefeuille (0-15 points de risque)
    if starting_equity > 0:
        dd_pct = ((starting_equity - current_equity) / starting_equity) * 100
        if dd_pct > 10:
            risk_score += 15
        elif dd_pct > 5:
            risk_score += 10
        elif dd_pct > 0:
            risk_score += 5
    
    return min(100, max(0, risk_score))

def calculate_position_size(cash: float, price: float, indicators: dict, 
                           current_equity: float, starting_equity: float) -> float:
    """
    Calcule la taille de position dynamique basée sur un score de risque multi-facteurs.
    Plus le risque est élevé, plus la position est petite.
    Minimum absolu: 10€ par position.
    """
    MIN_POSITION_VALUE = 10.0
    
    if cash < MIN_POSITION_VALUE * 1.1:
        return 0.0
    
    # Calculer le score de risque (0-100)
    risk_score = calculate_risk_score(indicators, current_equity, starting_equity)
    
    # Convertir le score de risque en pourcentage du capital à investir
    # Risque faible (0-30) : 50-70% du capital
    # Risque moyen (30-60) : 30-50% du capital
    # Risque élevé (60-100) : 10-30% du capital
    if risk_score <= 30:
        # Risque faible : investir beaucoup
        invest_pct = 0.70 - (risk_score / 30) * 0.20  # 70% à 50%
    elif risk_score <= 60:
        # Risque moyen : investir modérément
        invest_pct = 0.50 - ((risk_score - 30) / 30) * 0.20  # 50% à 30%
    else:
        # Risque élevé : investir peu
        invest_pct = 0.30 - ((risk_score - 60) / 40) * 0.20  # 30% à 10%
    
    # Calculer la taille de position
    position_value = cash * invest_pct
    
    # Limiter à 95% du cash maximum (garder un peu de marge)
    position_value = min(position_value, cash * 0.95)
    
    # Vérifier le minimum
    if position_value < MIN_POSITION_VALUE:
        if cash >= MIN_POSITION_VALUE * 1.1:
            return MIN_POSITION_VALUE
        else:
            return 0.0
    
    return position_value


def check_stop_loss(entry: float, current: float, atr: Optional[float], 
                    stop_loss_pct: float = 4.0, atr_multiplier: float = 2.0,
                    indicators: Optional[dict] = None, anti_panic: bool = True) -> Tuple[bool, str]:
    """
    Vérifie si le stop-loss est déclenché.
    Utilise le plus restrictif entre % fixe et ATR dynamique.
    
    ANTI-PANIC: Si le marché est en survente extrême, désactive le stop-loss fixe
    pour éviter de vendre au pire moment (principe contrarian).
    """
    loss_pct = ((entry - current) / entry) * 100
    
    if anti_panic and indicators:
        oversold_signals = []
        
        # RSI très survendu
        if indicators.get("rsi") and indicators["rsi"] < 25:
            oversold_signals.append(f"RSI={indicators['rsi']:.1f}")
        
        # Stochastic très survendu
        if indicators.get("stoch") and indicators["stoch"] < 20:
            oversold_signals.append(f"Stoch={indicators['stoch']:.1f}")
        
        # Prix sous Bollinger inférieure
        bb_lower = indicators.get("bb_lower")
        if bb_lower and current < bb_lower:
            oversold_signals.append("Prix<BB_inf")
        
        # Si au moins 2 signaux de survente, désactiver le stop-loss fixe
        if len(oversold_signals) >= 2:
            print(f"   🛡️  ANTI-PANIC: Stop-loss fixe désactivé (survente détectée: {', '.join(oversold_signals)})", flush=True)
            # On garde seulement le stop-loss ATR qui est plus large
            if atr is not None and atr > 0:
                atr_stop = atr * atr_multiplier
                if (entry - current) >= atr_stop:
                    return True, f"Stop-loss ATR déclenché (perte {entry - current:.4f} >= {atr_stop:.4f})"
            return False, ""
    
    # Stop-loss fixe normal
    if loss_pct >= stop_loss_pct:
        return True, f"Stop-loss fixe déclenché ({loss_pct:.2f}% >= {stop_loss_pct}%)"
    
    # Stop-loss dynamique basé sur ATR
    if atr is not None and atr > 0:
        atr_stop = atr * atr_multiplier
        if (entry - current) >= atr_stop:
            return True, f"Stop-loss ATR déclenché (perte {entry - current:.4f} >= {atr_stop:.4f})"
    
    return False, ""

def check_take_profit(entry: float, current: float, atr: Optional[float],
                     take_profit_pct: float = 3.0, atr_multiplier: float = 3.0) -> Tuple[bool, str]:
    """
    Vérifie si le take-profit est atteint.
    Utilise le moins restrictif entre % fixe et ATR dynamique.
    """
    profit_pct = ((current - entry) / entry) * 100
    
    # Take-profit fixe
    if profit_pct >= take_profit_pct:
        return True, f"Take-profit fixe atteint ({profit_pct:.2f}% >= {take_profit_pct}%)"
    
    # Take-profit dynamique basé sur ATR
    if atr is not None and atr > 0:
        atr_target = atr * atr_multiplier
        if (current - entry) >= atr_target:
            return True, f"Take-profit ATR atteint (gain {current - entry:.4f} >= {atr_target:.4f})"
    
    return False, ""

def check_trailing_stop(entry: float, current: float, highest: float, 
                       trailing_pct: float = 1.5) -> Tuple[bool, str]:
    """
    Trailing stop: vend si le prix baisse de trailing_pct% depuis le plus haut.
    """
    if highest <= entry:
        return False, ""
    
    drop_from_high = ((highest - current) / highest) * 100
    
    if drop_from_high >= trailing_pct:
        profit_pct = ((current - entry) / entry) * 100
        return True, f"Trailing stop déclenché (baisse {drop_from_high:.2f}% depuis sommet, profit actuel {profit_pct:.2f}%)"
    
    return False, ""

def check_quick_profit_lock(entry: float, current: float, highest: float,
                           profit_threshold: float = 0.5, drop_threshold: float = 0.3) -> Tuple[bool, str]:
    """
    Quick profit lock: sécurise rapidement les petits gains.
    Si gain > profit_threshold% ET prix baisse de drop_threshold% depuis le pic, alors vendre.
    """
    current_profit_pct = ((current - entry) / entry) * 100
    
    # Vérifier si on a atteint le seuil de profit minimum
    if current_profit_pct < profit_threshold:
        return False, ""
    
    # Vérifier si le prix a baissé depuis le plus haut
    if highest <= entry:
        return False, ""
    
    drop_from_high = ((highest - current) / highest) * 100
    
    if drop_from_high >= drop_threshold:
        return True, f"Quick profit lock déclenché (gain {current_profit_pct:.2f}%, baisse {drop_from_high:.2f}% depuis pic)"
    
    return False, ""

def check_safety_score_sell(entry: float, current: float, qty: float, platform: str,
                            indicators: dict, highest: float, entry_fees: float) -> Tuple[bool, str]:
    """
    Vérifie si on doit vendre basé sur le score de sûreté.
    S'active uniquement si on est en profit NET (tous frais compris).
    Vend si le score de sûreté < 60/100.
    """
    # Calculer le profit NET en incluant TOUS les frais
    buy_amount = qty * entry
    total_buy_cost = buy_amount + entry_fees
    
    # Estimer les frais de vente
    sell_amount = qty * current
    sell_fees = calculate_realistic_fees(sell_amount, qty, current, "SELL", platform)
    net_proceeds = sell_amount - sell_fees["total"]
    
    # Profit NET = ce qu'on récupère - ce qu'on a investi
    net_profit = net_proceeds - total_buy_cost
    net_profit_pct = (net_profit / total_buy_cost) * 100
    
    # Vérifier si on est en profit NET
    if net_profit <= 0:
        return False, ""
    
    # Calculer le score de sûreté
    safety_score = calculate_safety_score(indicators, current, entry, highest)
    
    # Vendre si le score est inférieur à 60
    if safety_score < 60:
        return True, f"Score de sûreté faible ({safety_score:.0f}/100) avec profit NET {net_profit:+.2f}€ ({net_profit_pct:+.2f}%)"
    
    return False, ""


# ========= Clés API =========
def load_api_keys_from_env() -> List[str]:
    keys = []
    keys_csv = os.getenv("FINNHUB_KEYS", "").strip()
    if keys_csv:
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    else:
        for name, val in os.environ.items():
            if name.upper().startswith("FINNHUB_API_KEY") and val.strip():
                keys.append(val.strip())
        keys = [os.getenv("FINNHUB_API_KEY", "")] + \
               [os.getenv(f"FINNHUB_API_KEY{i}", "") for i in range(2, 10)]
        keys = [k for k in keys if k]
    return keys

# ========= Rate limiters optimisés =========
class KeyPacer:
    """Rate-limiter optimisé pour 60 req/min par clé."""
    def __init__(self, api_keys: List[str], max_rpm_per_key: int = 60):
        self.api_keys = api_keys
        self.n = len(api_keys)
        self.max_rpm = max_rpm_per_key
        self.backoff_until: Dict[int, float] = {}
        self.errors: Dict[int, int] = {}
        self.reqs_window: Dict[int, deque] = {i: deque() for i in range(self.n)}
        self.next_key_idx_for_ticker = defaultdict(int)

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _purge_old(self, idx: int) -> None:
        win = self.reqs_window[idx]
        now = self._now()
        while win and (now - win[0]) > 60.0:
            win.popleft()

    def has_capacity(self, idx: int) -> bool:
        self._purge_old(idx)
        return len(self.reqs_window[idx]) < self.max_rpm

    def record_attempt(self, idx: int) -> None:
        self._purge_old(idx)
        self.reqs_window[idx].append(self._now())

    def start_backoff(self, idx: int) -> int:
        self.errors[idx] = self.errors.get(idx, 0) + 1
        backoff = min(60, 2 ** self.errors[idx])
        self.backoff_until[idx] = time.time() + backoff
        return backoff

    def clear_error(self, idx: int) -> None:
        self.errors[idx] = 0

    def key_available(self, idx: int) -> bool:
        return (time.time() >= self.backoff_until.get(idx, 0.0)) and self.has_capacity(idx)

    def choose_key_for(self, sym: str) -> Tuple[str, int]:
        if self.n == 0:
            raise RuntimeError("Aucune clé API disponible")
        start = self.next_key_idx_for_ticker[sym] % self.n
        for k in range(self.n):
            idx = (start + k) % self.n
            if self.key_available(idx):
                self.next_key_idx_for_ticker[sym] = (idx + 1) % self.n
                return self.api_keys[idx], idx
        for k in range(self.n):
            idx = (start + k) % self.n
            if time.time() >= self.backoff_until.get(idx, 0.0):
                self.next_key_idx_for_ticker[sym] = (idx + 1) % self.n
                return self.api_keys[idx], idx
        idx = start
        return self.api_keys[idx], idx

class GlobalPacer:
    """Limiteur global optimisé."""
    def __init__(self, max_rpm_total: int = 60):
        self.max_rpm_total = max_rpm_total
        self.win = deque()

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _purge(self):
        now = self._now()
        while self.win and (now - self.win[0]) > 60.0:
            self.win.popleft()

    def has_capacity(self) -> bool:
        self._purge()
        return len(self.win) < self.max_rpm_total

    def record_attempt(self) -> None:
        self._purge()
        self.win.append(self._now())

# ========= Simulation avancée =========

def calculate_realistic_fees(amount: float, qty: float, price: float, side: str,
                            platform: str = "alpaca") -> dict:
    """
    Calcule les frais réalistes selon la plateforme de trading.
    
    Plateformes supportées:
    - alpaca: Frais Alpaca 2025 (commission-free + frais réglementaires)
    - degiro: Frais Degiro (0.50€ + 0.04%)
    - generic: Frais génériques configurables
    
    Args:
        amount: Montant de la transaction en €
        qty: Nombre d'actions
        price: Prix par action
        side: "BUY" ou "SELL"
        platform: Plateforme de trading
    
    Returns:
        dict avec détails des frais
    """
    if platform == "alpaca":
        # Frais Alpaca 2025 (en USD, convertis en EUR pour simplification)
        # CAT Fee: $0.0000265 par action (achat et vente)
        # FINRA TAF: $0.000166 par action (vente uniquement, max $8.30)
        
        cat_fee_per_share = 0.0000265
        finra_taf_per_share = 0.000166
        finra_taf_max = 8.30
        
        # CAT Fee (toujours appliqué)
        cat_fee = qty * cat_fee_per_share
        
        # FINRA TAF (uniquement à la vente)
        if side == "SELL":
            finra_taf = min(qty * finra_taf_per_share, finra_taf_max)
        else:
            finra_taf = 0.0
        
        total_fee = cat_fee + finra_taf
        
        return {
            "fixed": 0.0,
            "variable": cat_fee,
            "regulatory": finra_taf,
            "spread": 0.0,
            "total": total_fee,
            "total_pct": (total_fee / amount * 100) if amount > 0 else 0,
            "platform": "Alpaca"
        }
    
    elif platform == "degiro":
        # Frais Degiro: 0.50€ + 0.04%
        fee_fixed = 0.50
        fee_variable_bps = 4
        fee_var = amount * (fee_variable_bps / 10000.0)
        total_fee = fee_fixed + fee_var
        
        return {
            "fixed": fee_fixed,
            "variable": fee_var,
            "regulatory": 0.0,
            "spread": 0.0,
            "total": total_fee,
            "total_pct": (total_fee / amount * 100) if amount > 0 else 0,
            "platform": "Degiro"
        }
    
    else:  # generic
        # Frais génériques configurables
        fee_fixed = float(os.getenv("FEE_FIXED", "0.50"))
        fee_variable_bps = float(os.getenv("FEE_VARIABLE_BPS", "4"))
        spread_bps = float(os.getenv("SPREAD_BPS", "5"))
        
        fee_var = amount * (fee_variable_bps / 10000.0)
        spread_cost = amount * (spread_bps / 10000.0)
        total_fee = fee_fixed + fee_var + spread_cost
        
        return {
            "fixed": fee_fixed,
            "variable": fee_var,
            "regulatory": 0.0,
            "spread": spread_cost,
            "total": total_fee,
            "total_pct": (total_fee / amount * 100) if amount > 0 else 0,
            "platform": "Generic"
        }

def sim_buy(engine, cash_mem: dict, symbol: str, price: float, indicators: dict,
            platform: str, position_tracker: dict, current_equity: float, starting_equity: float):
    if get_position(engine, symbol):
        return
    
    # Position sizing dynamique basé sur le score de risque
    size = calculate_position_size(cash_mem["cash"], price, indicators, current_equity, starting_equity)
    
    if size < 10.0:
        if size > 0:
            print(f"[{ts_str()}] ⚠️  Position trop petite ({size:.2f}€ < 10€), achat annulé", flush=True)
        return
    
    # Calculer le score de risque pour l'affichage
    risk_score = calculate_risk_score(indicators, current_equity, starting_equity)
    
    # Calculer la quantité d'actions
    qty = size / price
    
    # Calculer les frais réalistes Alpaca
    fees = calculate_realistic_fees(size, qty, price, "BUY", platform)
    total_cost = size + fees["total"]
    
    # Vérifier qu'on a assez de cash
    if total_cost > cash_mem["cash"]:
        print(f"[{ts_str()}] ⚠️  Cash insuffisant ({cash_mem['cash']:.2f}€ < {total_cost:.2f}€)", flush=True)
        return
    
    upsert_position_buy(engine, symbol, qty, price)
    insert_trade(engine, utcnow(), symbol, "BUY", qty, price, fees["total"])
    cash_mem["cash"] = round(cash_mem["cash"] - total_cost, 8)
    
    position_tracker[symbol] = {
        "entry": price,
        "highest": price,
        "entry_time": utcnow(),
        "entry_fees": fees["total"]  # Stocker les frais d'achat
    }
    
    print(f"[{ts_str()}] 💰 Achat confirmé: {size:.2f}€ investi ({qty:.6f} parts @ {price:.4f})", flush=True)
    if platform == "alpaca":
        print(f"[{ts_str()}]    Frais {fees['platform']}: CAT {fees['variable']:.4f}€ = {fees['total']:.4f}€ ({fees['total_pct']:.3f}%)", flush=True)
    else:
        print(f"[{ts_str()}]    Frais {fees['platform']}: {fees['fixed']:.2f}€ fixe + {fees['variable']:.2f}€ variable = {fees['total']:.2f}€ ({fees['total_pct']:.2f}%)", flush=True)
    print(f"[{ts_str()}]    Score de risque: {risk_score:.1f}/100 ({size/cash_mem['cash']*100:.1f}% du capital)", flush=True)

def sim_sell(engine, cash_mem: dict, symbol: str, price: float, 
             platform: str, position_tracker: dict):
    pos = get_position(engine, symbol)
    if not pos:
        return
    qty, entry = pos
    
    gross = qty * price
    
    # Calculer les frais réalistes Alpaca
    fees = calculate_realistic_fees(gross, qty, price, "SELL", platform)
    
    net = gross - fees["total"]
    
    delete_position(engine, symbol)
    insert_trade(engine, utcnow(), symbol, "SELL", qty, price, fees["total"])
    cash_mem["cash"] = round(cash_mem["cash"] + net, 8)
    
    # Calculer le PnL NET (incluant les frais d'achat)
    buy_amount = qty * entry
    entry_fees = position_tracker.get(symbol, {}).get("entry_fees", 0.0)
    if entry_fees == 0.0:
        # Estimer les frais d'achat si non disponibles
        buy_fees = calculate_realistic_fees(buy_amount, qty, entry, "BUY", platform)
        entry_fees = buy_fees["total"]
    
    total_buy_cost = buy_amount + entry_fees
    
    pnl_net = net - total_buy_cost
    pnl_pct = ((price - entry) / entry) * 100
    pnl_net_pct = (pnl_net / total_buy_cost) * 100
    
    print(f"[{ts_str()}] 💸 Vente confirmée: {qty:.6f} parts @ {price:.4f} = {gross:.2f}€", flush=True)
    if platform == "alpaca":
        print(f"[{ts_str()}]    Frais {fees['platform']}: CAT {fees['variable']:.4f}€ + FINRA {fees['regulatory']:.4f}€ = {fees['total']:.4f}€ ({fees['total_pct']:.3f}%)", flush=True)
    else:
        print(f"[{ts_str()}]    Frais {fees['platform']}: {fees['fixed']:.2f}€ fixe + {fees['variable']:.2f}€ variable = {fees['total']:.2f}€ ({fees['total_pct']:.2f}%)", flush=True)
    print(f"[{ts_str()}]    PnL brut: {(price - entry) * qty:+.2f}€ ({pnl_pct:+.2f}%)", flush=True)
    print(f"[{ts_str()}]    PnL NET (frais compris): {pnl_net:+.2f}€ ({pnl_net_pct:+.2f}%)", flush=True)
    
    # Nettoyer le tracker
    if symbol in position_tracker:
        del position_tracker[symbol]

def emergency_sell_all(engine, cash_mem: dict, position_tracker: dict, 
                      platform: str, api_key: str) -> None:
    """
    🚨 VENTE D'URGENCE: Liquide toutes les positions immédiatement.
    Utilisé en cas de crash du marché ou de problème critique.
    """
    print(f"\n{'='*120}", flush=True)
    print(f"🚨 ALERTE: VENTE D'URGENCE DÉCLENCHÉE", flush=True)
    print(f"{'='*120}\n", flush=True)
    
    with engine.begin() as con:
        positions = list(con.execute(text("select symbol, qty, entry from positions")))
    
    if not positions:
        print(f"[{ts_str()}] ℹ️  Aucune position à liquider.", flush=True)
        return
    
    total_pnl = 0.0
    total_fees = 0.0
    sold_count = 0
    
    for symbol, qty, entry in positions:
        try:
            # Récupérer le prix actuel
            q = fetch_quote(symbol, api_key)
            price = q.get("c")
            
            if not price or price <= 0:
                print(f"[{ts_str()}] ⚠️  {symbol}: Prix invalide, skip", flush=True)
                continue
            
            price = float(price)
            
            gross = qty * price
            
            # Calculer les frais
            fees = calculate_realistic_fees(gross, qty, price, "SELL", platform)
            net = gross - fees["total"]
            
            # Calculer le PnL
            pnl = net - (qty * entry)
            pnl_pct = ((price - entry) / entry) * 100
            total_pnl += pnl
            total_fees += fees["total"]
            
            # Vendre
            sim_sell(engine, cash_mem, symbol, price, platform, position_tracker)
            sold_count += 1
            
            print(f"[{ts_str()}] 🔴 {symbol} VENDU: {qty:.6f} @ {price:.4f} | PnL: {pnl:+.2f}€ ({pnl_pct:+.2f}%) | Frais: {fees['total']:.4f}€", flush=True)
            
            time.sleep(0.5)  # Petit délai entre les ventes
            
        except Exception as e:
            print(f"[{ts_str()}] ❌ {symbol}: Erreur lors de la vente d'urgence: {e}", flush=True)
    
    print(f"\n{'='*120}", flush=True)
    print(f"✅ VENTE D'URGENCE TERMINÉE", flush=True)
    print(f"   Positions liquidées: {sold_count}/{len(positions)}", flush=True)
    print(f"   PnL total: {total_pnl:+.2f}€", flush=True)
    print(f"   Frais totaux: {total_fees:.4f}€", flush=True)
    print(f"   Cash disponible: {cash_mem['cash']:.2f}€", flush=True)
    print(f"{'='*120}\n", flush=True)

def check_emergency_trigger() -> bool:
    """
    Vérifie si une vente d'urgence doit être déclenchée.
    Retourne True si :
    - Le fichier /tmp/emergency_sell existe
    - La variable d'environnement EMERGENCY_SELL=true
    """
    # Vérifier le fichier trigger
    if os.path.exists("/tmp/emergency_sell"):
        return True
    
    # Vérifier la variable d'environnement
    if os.getenv("EMERGENCY_SELL", "").lower() == "true":
        return True
    
    return False

def calculate_safety_score(indicators: dict, current: float, entry: float, highest: float) -> float:
    """
    Calcule un score de sûreté 0-100. Plus le score est élevé, plus c'est sûr de vendre.
    S'active uniquement si on est en profit NET.
    """
    score = 0.0
    factors = 0
    
    # Facteur 1: Gain actuel (0-30 points)
    if current > entry:
        profit_pct = ((current - entry) / entry) * 100
        if profit_pct > 5:
            score += 30
        elif profit_pct > 2:
            score += 15
        elif profit_pct > 0.5:
            score += 5
        factors += 1
    
    # Facteur 2: Baisse depuis le plus haut (0-25 points)
    if highest > entry and current < highest:
        drop_from_high = ((highest - current) / highest) * 100
        if drop_from_high > 2.0:
            score += 25
        elif drop_from_high > 1.0:
            score += 15
        elif drop_from_high > 0.5:
            score += 5
        factors += 1
    
    # Facteur 3: RSI suracheté (0-15 points)
    rsi = indicators.get("rsi")
    if rsi and rsi > 70:
        score += 15
    elif rsi and rsi > 60:
        score += 7
    if rsi: factors += 1
    
    # Facteur 4: MACD baissier (0-15 points)
    macd_hist = indicators.get("macd_hist")
    if macd_hist and macd_hist < -0.5:
        score += 15
    elif macd_hist and macd_hist < -0.2:
        score += 7
    if macd_hist: factors += 1
    
    # Facteur 5: Stochastique suracheté (0-15 points)
    stoch = indicators.get("stoch")
    if stoch and stoch > 80:
        score += 15
    elif stoch and stoch > 70:
        score += 7
    if stoch: factors += 1
    
    # Facteur 6: Volatilité élevée (0-10 points)
    volatility = indicators.get("volatility_regime", "normal")
    if volatility == "high":
        score += 10
    elif volatility == "normal":
        score += 5
    factors += 1
    
    # Normaliser le score
    if factors > 0:
        return min(100, (score / factors) * (100 / 20)) # Normaliser à 100
    return 50.0

def check_max_drawdown(current_equity: float, starting_equity: float, max_drawdown_pct: float) -> Tuple[bool, str]:
    """
    Vérifie si le drawdown maximum du portefeuille a été dépassé.
    Retourne (True, message) si le drawdown est dépassé, sinon (False, "").
    """
    if starting_equity <= 0:
        return False, ""
    
    drawdown = ((starting_equity - current_equity) / starting_equity) * 100
    
    if drawdown >= max_drawdown_pct:
        return True, f"Drawdown du portefeuille atteint ({drawdown:.2f}% >= {max_drawdown_pct}%)"
    
    return False, ""

def equity_total(engine, last_prices: Dict[str, float], cash: float) -> float:
    """
    Calcule l'équité totale du portefeuille (cash + valeur des positions ouvertes).
    """
    with engine.begin() as con:
        positions = list(con.execute(text("select symbol, qty, entry from positions")))
    
    total_value = cash
    for symbol, qty, entry in positions:
        price = last_prices.get(symbol)
        if price is not None and price > 0:
            total_value += qty * price
        else:
            # Si le prix n'est pas disponible, utiliser le prix d'entrée (moins précis)
            total_value += qty * entry
            
    return total_value

# ========= Main loop =========
def main():
    load_dotenv()

    api_keys = load_api_keys_from_env()
    if not api_keys:
        raise SystemExit("Aucune clé API Finnhub fournie.")
    print(f"🔑 Clés Finnhub: {len(api_keys)} (optimisé pour 60 req/min)", flush=True)

    db_url = os.getenv("DATABASE_URL", "postgresql://bot:botpass@db:5433/trading")
    tickers = [t.strip().upper() for t in os.getenv("TICKERS", "AAPL,TSLA,NVDA").split(",") if t.strip()]

    # Paramètres optimisés pour 60 req/min
    sleep_s = float(os.getenv("SLEEP_BETWEEN_CALLS_SECONDS", "0.5"))
    closed_sleep = float(os.getenv("CLOSED_SLEEP_SECONDS", "60"))
    sim_mode = os.getenv("SIM_MODE", "true").lower() == "true"
    start = float(os.getenv("STARTING_CASH", "100"))
    
    platform = os.getenv("TRADING_PLATFORM", "alpaca").lower()  # alpaca, degiro, generic
    
    fast_mode = os.getenv("FAST_MODE", "false").lower() == "true"

    active_hours_str = os.getenv("ACTIVE_HOURS", "9-17.5")
    active_hours = parse_active_hours(active_hours_str)

    stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "4.0"))
    take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))
    trailing_stop_pct = float(os.getenv("TRAILING_STOP_PCT", "1.5"))
    min_confidence = float(os.getenv("MIN_CONFIDENCE_SCORE", "60.0"))
    max_drawdown_pct = float(os.getenv("MAX_DRAWDOWN_PCT", "15.0"))

    quick_profit_enabled = os.getenv("QUICK_PROFIT_ENABLED", "true").lower() == "true"
    quick_profit_threshold = float(os.getenv("QUICK_PROFIT_THRESHOLD", "0.5"))
    quick_profit_drop = float(os.getenv("QUICK_PROFIT_DROP", "0.3"))
    
    anti_panic_enabled = os.getenv("ANTI_PANIC_ENABLED", "true").lower() == "true"
    
    safety_score_enabled = os.getenv("SAFETY_SCORE_ENABLED", "true").lower() == "true"

    # Rate limits optimisés
    max_rpm_per_key = int(os.getenv("FINNHUB_MAX_RPM", "60"))
    max_rpm_total = int(os.getenv("FINNHUB_MAX_RPM_TOTAL", "60"))
    ticker_min_refresh = float(os.getenv("TICKER_MIN_REFRESH_SECONDS", "30"))
    global_cooldown_sec = int(os.getenv("FINNHUB_GLOBAL_COOLDOWN_SECONDS", "60"))

    engine = create_engine(db_url, pool_pre_ping=True)
    cash_mem = {"cash": start}
    last_trade_at: Dict[str, datetime] = {}
    position_tracker: Dict[str, dict] = {}
    last_prices: Dict[str, float] = {}

    pacer = KeyPacer(api_keys, max_rpm_per_key=max_rpm_per_key)
    global_pacer = GlobalPacer(max_rpm_total=max_rpm_total)

    last_quote_at: Dict[str, float] = {}
    quota_exhausted_until = 0.0

    profile = "ULTRA-FAST" if fast_mode else "STANDARD"
    print(f"{'='*120}", flush=True)
    print(f"🚀 BOT DE TRADING AVANCÉ - {profile}", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"📊 Tickers: {', '.join(tickers)}", flush=True)
    print(f"💰 Capital initial: {start:.2f} €", flush=True)
    print(f"💳 Plateforme: {platform.upper()}", flush=True)
    
    if platform == "alpaca":
        print(f"   Frais Alpaca 2025 (commission-free):", flush=True)
        print(f"   • CAT Fee: $0.0000265 par action (achat et vente)", flush=True)
        print(f"   • FINRA TAF: $0.000166 par action (vente uniquement, max $8.30)", flush=True)
        # Exemple sur 100 actions à 50€
        example_qty = 100
        example_price = 50.0
        example_amount = example_qty * example_price
        buy_fees = calculate_realistic_fees(example_amount, example_qty, example_price, "BUY", "alpaca")
        sell_fees = calculate_realistic_fees(example_amount, example_qty, example_price, "SELL", "alpaca")
        print(f"   • Exemple 100 actions @ 50€: Achat {buy_fees['total']:.4f}€, Vente {sell_fees['total']:.4f}€, Total {buy_fees['total'] + sell_fees['total']:.4f}€", flush=True)
    else:
        print(f"   Frais de trading: voir configuration", flush=True)
    
    print(f"🛡️  Protections:", flush=True)
    print(f"   • Stop-loss: {stop_loss_pct}%", flush=True)
    print(f"   • Take-profit: {take_profit_pct}%", flush=True)
    print(f"   • Trailing stop: {trailing_stop_pct}%", flush=True)
    if quick_profit_enabled:
        print(f"   • Quick profit lock: gain >{quick_profit_threshold}% puis baisse >{quick_profit_drop}%", flush=True)
    if safety_score_enabled:
        print(f"   • Score de sûreté: vend si score <60/100 en profit NET", flush=True)
    if anti_panic_enabled:
        print(f"   • Anti-panic: désactive stop-loss en survente extrême", flush=True)
    print(f"   • Max drawdown: {max_drawdown_pct}%", flush=True)
    print(f"   • Position sizing: dynamique basé sur score de risque (min 10€)", flush=True)
    print(f"   • Confiance minimale: {min_confidence:.0f}/100", flush=True)
    print(f"⚡ Rate limiting: {max_rpm_total} req/min global, {max_rpm_per_key} req/min par clé", flush=True)
    print(f"🕐 Horaires: {active_hours_str} (Europe/Paris)", flush=True)
    print(f"🚨 Vente d'urgence: créer /tmp/emergency_sell ou EMERGENCY_SELL=true", flush=True)
    print(f"{'='*120}", flush=True)

    print(f"\n[{ts_str()}] 🧹 Nettoyage des positions au démarrage...", flush=True)
    emergency_sell_all(engine, cash_mem, position_tracker, platform, api_keys[0])
    print(f"[{ts_str()}] ✅ Nettoyage terminé, démarrage du bot...\n", flush=True)
    time.sleep(2)

    while True:
        try:
            if check_emergency_trigger():
                emergency_sell_all(engine, cash_mem, position_tracker, platform, api_keys[0])
                print(f"[{ts_str()}] ⏸️  Pause de 60s après vente d'urgence...", flush=True)
                time.sleep(60)
                continue

            if time.time() < quota_exhausted_until:
                time.sleep(0.5)
                continue

            is_open, reason = within_paris_session(active_hours)
            if not is_open:
                print(f"[{ts_str()}] ⏸️  Marché fermé — {reason}", flush=True)
                time.sleep(closed_sleep)
                continue

            current_eq = equity_total(engine, last_prices, cash_mem["cash"])
            dd_triggered, dd_msg = check_max_drawdown(current_eq, start, max_drawdown_pct)
            if dd_triggered:
                print(f"[{ts_str()}] 🛑 ARRÊT D'URGENCE: {dd_msg}", flush=True)
                emergency_sell_all(engine, cash_mem, position_tracker, platform, api_keys[0])
                time.sleep(300)
                continue

            for sym in tickers:
                try:
                    now_mono = time.monotonic()
                    last_mono = last_quote_at.get(sym, 0.0)
                    if (now_mono - last_mono) < ticker_min_refresh:
                        time.sleep(0.02)
                        continue
                    last_quote_at[sym] = now_mono

                    api_key, key_idx = pacer.choose_key_for(sym)
                    active_fp = key_fp(api_key)

                    now_epoch = time.time()
                    until = pacer.backoff_until.get(key_idx, 0.0)
                    if now_epoch < until:
                        time.sleep(0.05)
                        continue

                    if not pacer.has_capacity(key_idx):
                        time.sleep(0.1)
                        continue

                    if not global_pacer.has_capacity():
                        time.sleep(0.1)
                        continue

                    global_pacer.record_attempt()
                    pacer.record_attempt(key_idx)
                    
                    try:
                        q = fetch_quote(sym, api_key)
                    except requests.HTTPError as e:
                        status = e.response.status_code if e.response is not None else None
                        body = e.response.text if e.response is not None else str(e)

                        if status == 429:
                            backoff = pacer.start_backoff(key_idx)
                            print(f"[{ts_str()}] {sym} (key#{key_idx+1}) 429 → backoff {backoff}s", flush=True)

                            if "Remaining Limit: 0" in (body or ""):
                                quota_exhausted_until = time.time() + global_cooldown_sec
                                print(f"[{ts_str()}] ⚡ Quota épuisé → pause {global_cooldown_sec}s", flush=True)

                            time.sleep(0.2)
                            continue
                        else:
                            print(f"[{ts_str()}] {sym} HTTPError {status}", flush=True)
                            time.sleep(0.8)
                            continue
                    else:
                        pacer.clear_error(key_idx)

                    price = q.get("c")
                    if not price or price <= 0:
                        time.sleep(sleep_s)
                        continue
                    price = float(price)
                    
                    last_prices[sym] = price

                    now_ts = utcnow().replace(microsecond=0)
                    insert_tick(engine, sym, now_ts, price)

                    ind = compute_advanced_indicators(engine, sym, fast_mode)
                    if not ind:
                        print(f"[{ts_str()}] {sym} en attente d'historique...", flush=True)
                        time.sleep(sleep_s)
                        continue

                    # Extraction des indicateurs
                    close = ind["close"]
                    rsi = ind["rsi"]
                    macd_val = ind["macd"]
                    macd_sig = ind["macd_signal"]
                    macd_hist = ind["macd_hist"]
                    bb_upper = ind["bb_upper"]
                    bb_lower = ind["bb_lower"]
                    atr_val = ind["atr"]
                    stoch = ind["stoch"]
                    confidence = ind["confidence_score"]
                    volatility = ind["volatility_regime"]
                    mtf_score = ind["mtf_score"] # Extraire le score MTF

                    pos = get_position(engine, sym)
                    if pos:
                        qty, entry = pos
                        
                        # Mettre à jour le plus haut pour trailing stop
                        if sym in position_tracker:
                            if price > position_tracker[sym]["highest"]:
                                position_tracker[sym]["highest"] = price
                            highest = position_tracker[sym]["highest"]
                            entry_fees = position_tracker[sym].get("entry_fees", 0.0)
                        else:
                            highest = price
                            entry_fees = 0.0
                            position_tracker[sym] = {"entry": entry, "highest": price, "entry_time": utcnow(), "entry_fees": 0.0}
                        
                        if safety_score_enabled:
                            safety_triggered, safety_msg = check_safety_score_sell(
                                entry, price, qty, platform, ind, highest, entry_fees
                            )
                            if safety_triggered:
                                print(f"[{ts_str()}] 🎯 {sym} {safety_msg}", flush=True)
                                sim_sell(engine, cash_mem, sym, price, platform, position_tracker)
                                time.sleep(sleep_s)
                                continue
                        
                        sl_triggered, sl_msg = check_stop_loss(entry, price, atr_val, stop_loss_pct, 
                                                               indicators=ind, anti_panic=anti_panic_enabled)
                        if sl_triggered:
                            print(f"[{ts_str()}] 🛑 {sym} {sl_msg}", flush=True)
                            sim_sell(engine, cash_mem, sym, price, platform, position_tracker)
                            time.sleep(sleep_s)
                            continue
                        
                        # Vérifier take-profit
                        tp_triggered, tp_msg = check_take_profit(entry, price, atr_val, take_profit_pct)
                        if tp_triggered:
                            print(f"[{ts_str()}] 🎯 {sym} {tp_msg}", flush=True)
                            sim_sell(engine, cash_mem, sym, price, platform, position_tracker)
                            time.sleep(sleep_s)
                            continue
                        
                        # Vérifier trailing stop
                        ts_triggered, ts_msg = check_trailing_stop(entry, price, highest, trailing_stop_pct)
                        if ts_triggered:
                            print(f"[{ts_str()}] 📉 {sym} {ts_msg}", flush=True)
                            sim_sell(engine, cash_mem, sym, price, platform, position_tracker)
                            time.sleep(sleep_s)
                            continue
                        
                        if quick_profit_enabled:
                            qpl_triggered, qpl_msg = check_quick_profit_lock(entry, price, highest, 
                                                                            quick_profit_threshold, quick_profit_drop)
                            if qpl_triggered:
                                print(f"[{ts_str()}] ⚡ {sym} {qpl_msg}", flush=True)
                                sim_sell(engine, cash_mem, sym, price, platform, position_tracker)
                                time.sleep(sleep_s)
                                continue
                        
                        # Vérifier conditions de vente (RSI suracheté ou MACD baissier)
                        if (rsi and rsi > 75) or (macd_hist and macd_hist < -0.5):
                            reason = "RSI>75" if (rsi and rsi > 75) else "MACD baissier"
                            print(f"[{ts_str()}] 📊 {sym} Signal SELL: {reason}", flush=True)
                            sim_sell(engine, cash_mem, sym, price, platform, position_tracker)
                            time.sleep(sleep_s)
                            continue

                    else:
                        # Conditions d'achat multiples
                        buy_signals = []
                        buy_reasons = []
                        
                        # Signal 1: RSI survendu
                        if rsi and rsi < 35:
                            buy_signals.append(True)
                            buy_reasons.append(f"RSI survendu ({rsi:.1f})")
                        
                        # Signal 2: MACD croisement haussier
                        if macd_val and macd_sig and macd_hist and macd_val > macd_sig and macd_hist > 0:
                            buy_signals.append(True)
                            buy_reasons.append("MACD haussier")
                        
                        # Signal 3: Prix proche de la bande de Bollinger inférieure
                        if bb_lower and price <= bb_lower * 1.02:
                            buy_signals.append(True)
                            buy_reasons.append("Prix près BB inférieure")
                        
                        # Signal 4: Stochastic survendu
                        if stoch and stoch < 25:
                            buy_signals.append(True)
                            buy_reasons.append(f"Stoch survendu ({stoch:.1f})")
                        
                        current_eq = equity_total(engine, last_prices, cash_mem["cash"])
                        risk_score = calculate_risk_score(ind, current_eq, start)
                        
                        # Décision d'achat: au moins 2 signaux + score de confiance suffisant + score MTF suffisant
                        if len(buy_signals) >= 2 and confidence >= min_confidence and mtf_score >= 70: # Added MTF score check
                            print(f"[{ts_str()}] ✅ {sym} Signal BUY (Confiance: {confidence:.1f}/100, MTF: {mtf_score:.0f}/100, Risque: {risk_score:.1f}/100)", flush=True)
                            print(f"[{ts_str()}]    Raisons: {', '.join(buy_reasons)}", flush=True)
                            print(f"[{ts_str()}]    Volatilité: {volatility}, ATR: {fmt(atr_val, 4)}", flush=True)
                            sim_buy(engine, cash_mem, sym, price, ind, platform, 
                                   position_tracker, current_eq, start)
                            last_trade_at[sym] = utcnow()

                    eq = equity_total(engine, last_prices, cash_mem["cash"])
                    insert_equity(engine, now_ts, eq)

                    print(f"{'─'*120}", flush=True)
                    print(f"[{ts_str()}] {sym:<6} | Prix: {fmt(price)} | Confiance: {confidence:.0f}/100 | MTF: {mtf_score:.0f}/100 | Vol: {volatility}", flush=True)
                    print(f"   RSI: {fmt(rsi,1)} | MACD: {fmt(macd_hist,3)} | Stoch: {fmt(stoch,1)} | ATR: {fmt(atr_val,4)}", flush=True)
                    if pos:
                        pnl = (price - entry) * qty
                        pnl_pct = ((price - entry) / entry) * 100
                        if safety_score_enabled and entry_fees > 0: # Ensure entry fees are available to calculate net profit
                            safety_score = calculate_safety_score(ind, price, entry, highest)
                            print(f"   💼 Position: {qty:.6f} @ {entry:.4f} | PnL: {pnl:+.2f}€ ({pnl_pct:+.2f}%) | Sûreté: {safety_score:.0f}/100", flush=True)
                        else:
                            print(f"   💼 Position: {qty:.6f} @ {entry:.4f} | PnL: {pnl:+.2f}€ ({pnl_pct:+.2f}%)", flush=True)
                    print(f"   💰 Cash: {cash_mem['cash']:.2f}€ | Équité: {eq:.2f}€ | Gain total: {eq-start:+.2f}€ ({((eq/start-1)*100):+.2f}%)", flush=True)

                except Exception as e:
                    print(f"[{ts_str()}] {sym} ERROR: {e}", flush=True)

                time.sleep(sleep_s)

        except Exception as e:
            print(f"[{ts_str()}] LOOP ERROR: {e}", flush=True)
            time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Arrêt du bot.", flush=True)
