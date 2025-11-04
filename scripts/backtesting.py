#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtesting Engine - Optimisé pour Raspberry Pi 3B
Teste les stratégies de trading sur données historiques.
"""
import os
import sys
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import requests

# Importer les fonctions du bot principal
sys.path.append(os.path.dirname(__file__))

def fetch_historical_data(symbol: str, api_key: str, days: int = 30) -> pd.DataFrame:
    """
    Récupère les données historiques depuis Finnhub.
    Optimisé pour minimiser les appels API.
    """
    end_ts = int(time.time())
    start_ts = end_ts - (days * 24 * 60 * 60)
    
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": "1",  # 1 minute
        "from": start_ts,
        "to": end_ts,
        "token": api_key
    }
    
    print(f"Téléchargement des données pour {symbol} ({days} jours)...", flush=True)
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        if data.get("s") != "ok":
            raise ValueError(f"Erreur API: {data}")
        
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["t"], unit="s"),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })
        
        df = df.set_index("timestamp").sort_index()
        print(f"✓ {len(df)} bougies chargées pour {symbol}", flush=True)
        return df
        
    except Exception as e:
        print(f"✗ Erreur lors du téléchargement: {e}", flush=True)
        return pd.DataFrame()

def load_historical_csv(filepath: str) -> pd.DataFrame:
    """
    Charge des données historiques depuis un fichier CSV.
    Format attendu: timestamp,open,high,low,close,volume
    """
    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        print(f"✓ {len(df)} bougies chargées depuis {filepath}", flush=True)
        return df
    except Exception as e:
        print(f"✗ Erreur lors du chargement du CSV: {e}", flush=True)
        return pd.DataFrame()

def calculate_indicators_backtest(df: pd.DataFrame, fast_mode: bool = False) -> pd.DataFrame:
    """
    Calcule tous les indicateurs techniques sur le DataFrame.
    Version optimisée pour le backtesting (calcul vectorisé).
    """
    if fast_mode:
        sma_s, sma_l, rsi_len = 5, 20, 7
        macd_fast, macd_slow, macd_sig = 6, 13, 5
        bb_len, atr_len, stoch_len = 10, 7, 7
    else:
        sma_s, sma_l, rsi_len = 10, 50, 14
        macd_fast, macd_slow, macd_sig = 12, 26, 9
        bb_len, atr_len, stoch_len = 20, 14, 14
    
    # SMA
    df["sma_s"] = df["close"].rolling(sma_s, min_periods=sma_s).mean()
    df["sma_l"] = df["close"].rolling(sma_l, min_periods=sma_l).mean()
    
    # EMA
    df["ema_s"] = df["close"].ewm(span=sma_s, adjust=False, min_periods=sma_s).mean()
    df["ema_l"] = df["close"].ewm(span=sma_l, adjust=False, min_periods=sma_l).mean()
    
    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/rsi_len, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=1/rsi_len, adjust=False, min_periods=1).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    
    # MACD
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=macd_sig, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(bb_len, min_periods=bb_len).mean()
    bb_std = df["close"].rolling(bb_len, min_periods=bb_len).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2.0)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2.0)
    
    # ATR
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=atr_len, adjust=False, min_periods=atr_len).mean()
    
    # Stochastic
    lowest_low = df["low"].rolling(stoch_len, min_periods=stoch_len).min()
    highest_high = df["high"].rolling(stoch_len, min_periods=stoch_len).max()
    df["stoch"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    df["stoch"] = df["stoch"].fillna(50)
    
    # Volatilité
    df["recent_high"] = df["close"].rolling(60, min_periods=1).max()
    
    return df

def calculate_confidence_score_backtest(row: pd.Series) -> float:
    """Version simplifiée du score de confiance pour le backtesting."""
    score = 0.0
    factors = 0
    
    # Alignement moyennes mobiles
    if not pd.isna(row["sma_s"]) and not pd.isna(row["sma_l"]):
        if row["sma_s"] > row["sma_l"] and row["ema_s"] > row["ema_l"]:
            score += 20
        factors += 1
    
    # RSI
    if not pd.isna(row["rsi"]):
        if 30 <= row["rsi"] <= 70:
            score += 15
        factors += 1
    
    # MACD
    if not pd.isna(row["macd_hist"]) and row["macd_hist"] > 0:
        score += 20
        factors += 1
    
    # Bollinger
    if not pd.isna(row["bb_upper"]) and not pd.isna(row["bb_lower"]):
        bb_range = row["bb_upper"] - row["bb_lower"]
        if bb_range > 0:
            position = (row["close"] - row["bb_lower"]) / bb_range
            if 0.3 <= position <= 0.7:
                score += 15
        factors += 1
    
    # Stochastic
    if not pd.isna(row["stoch"]) and 20 <= row["stoch"] <= 80:
        score += 15
        factors += 1
    
    if factors > 0:
        return min(100, (score / factors) * (100 / 20))
    return 50.0

class BacktestEngine:
    """
    Moteur de backtesting optimisé pour Raspberry Pi 3B.
    Simule le trading avec la même logique que le bot live.
    """
    
    def __init__(self, starting_cash: float = 100.0, platform: str = "alpaca",
                 stop_loss_pct: float = 4.0, take_profit_pct: float = 3.0,
                 trailing_stop_pct: float = 1.5, min_confidence: float = 60.0,
                 quick_profit_threshold: float = 0.5, quick_profit_drop: float = 0.3):
        
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.platform = platform
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.min_confidence = min_confidence
        self.quick_profit_threshold = quick_profit_threshold
        self.quick_profit_drop = quick_profit_drop
        
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def calculate_fees(self, amount: float, qty: float, side: str) -> float:
        """Calcule les frais selon la plateforme."""
        if self.platform == "alpaca":
            cat_fee = qty * 0.0000265
            finra_taf = min(qty * 0.000166, 8.30) if side == "SELL" else 0.0
            return cat_fee + finra_taf
        else:
            return 0.50 + (amount * 0.0004)
    
    def buy(self, symbol: str, price: float, timestamp: datetime, 
            indicators: dict, risk_score: float) -> bool:
        """Simule un achat."""
        if symbol in self.positions:
            return False
        
        # Position sizing basé sur le risque
        if risk_score <= 30:
            invest_pct = 0.70 - (risk_score / 30) * 0.20
        elif risk_score <= 60:
            invest_pct = 0.50 - ((risk_score - 30) / 30) * 0.20
        else:
            invest_pct = 0.30 - ((risk_score - 60) / 40) * 0.20
        
        position_value = self.cash * invest_pct
        position_value = min(position_value, self.cash * 0.95)
        
        if position_value < 10.0:
            return False
        
        qty = position_value / price
        fees = self.calculate_fees(position_value, qty, "BUY")
        total_cost = position_value + fees
        
        if total_cost > self.cash:
            return False
        
        self.cash -= total_cost
        self.positions[symbol] = {
            "qty": qty,
            "entry": price,
            "highest": price,
            "entry_time": timestamp,
            "entry_fees": fees
        }
        
        self.trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": "BUY",
            "qty": qty,
            "price": price,
            "fees": fees,
            "cash_after": self.cash
        })
        
        return True
    
    def sell(self, symbol: str, price: float, timestamp: datetime, reason: str) -> bool:
        """Simule une vente."""
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        qty = pos["qty"]
        entry = pos["entry"]
        entry_fees = pos["entry_fees"]
        
        gross = qty * price
        fees = self.calculate_fees(gross, qty, "SELL")
        net = gross - fees
        
        self.cash += net
        
        # Calculer le PnL NET
        buy_cost = qty * entry + entry_fees
        pnl_net = net - buy_cost
        pnl_pct = ((price - entry) / entry) * 100
        
        self.trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": "SELL",
            "qty": qty,
            "price": price,
            "fees": fees,
            "pnl_net": pnl_net,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "cash_after": self.cash
        })
        
        del self.positions[symbol]
        return True
    
    def check_sell_conditions(self, symbol: str, price: float, 
                             indicators: dict, timestamp: datetime) -> Tuple[bool, str]:
        """Vérifie toutes les conditions de vente."""
        if symbol not in self.positions:
            return False, ""
        
        pos = self.positions[symbol]
        entry = pos["entry"]
        highest = pos["highest"]
        
        # Mettre à jour le plus haut
        if price > highest:
            pos["highest"] = price
            highest = price
        
        # Stop-loss
        loss_pct = ((entry - price) / entry) * 100
        if loss_pct >= self.stop_loss_pct:
            return True, f"Stop-loss ({loss_pct:.2f}%)"
        
        # Take-profit
        profit_pct = ((price - entry) / entry) * 100
        if profit_pct >= self.take_profit_pct:
            return True, f"Take-profit ({profit_pct:.2f}%)"
        
        # Trailing stop
        if highest > entry:
            drop_from_high = ((highest - price) / highest) * 100
            if drop_from_high >= self.trailing_stop_pct:
                return True, f"Trailing stop ({drop_from_high:.2f}%)"
        
        # Quick profit lock
        if profit_pct >= self.quick_profit_threshold and highest > entry:
            drop_from_high = ((highest - price) / highest) * 100
            if drop_from_high >= self.quick_profit_drop:
                return True, f"Quick profit lock"
        
        # Signaux techniques
        rsi = indicators.get("rsi")
        macd_hist = indicators.get("macd_hist")
        
        if (rsi and rsi > 75) or (macd_hist and macd_hist < -0.5):
            return True, "Signal technique"
        
        return False, ""
    
    def run(self, df: pd.DataFrame, symbol: str, fast_mode: bool = False) -> dict:
        """
        Exécute le backtest sur les données historiques.
        Retourne les métriques de performance.
        """
        print(f"\n{'='*80}")
        print(f"BACKTEST: {symbol}")
        print(f"Période: {df.index[0]} à {df.index[-1]}")
        print(f"Capital initial: {self.starting_cash:.2f}€")
        print(f"{'='*80}\n")
        
        # Calculer les indicateurs
        print("Calcul des indicateurs...", flush=True)
        df = calculate_indicators_backtest(df, fast_mode)
        
        # Supprimer les NaN au début
        df = df.dropna(subset=["sma_l", "rsi", "macd", "bb_lower", "atr"])
        
        print(f"Simulation sur {len(df)} bougies...\n", flush=True)
        
        # Simulation tick par tick
        for idx, row in df.iterrows():
            price = row["close"]
            
            # Calculer le score de confiance
            confidence = calculate_confidence_score_backtest(row)
            
            # Calculer le score de risque
            current_equity = self.cash + sum(
                pos["qty"] * price for pos in self.positions.values()
            )
            risk_score = self.calculate_risk_score(row, current_equity)
            
            indicators = {
                "rsi": row["rsi"],
                "macd": row["macd"],
                "macd_hist": row["macd_hist"],
                "bb_upper": row["bb_upper"],
                "bb_lower": row["bb_lower"],
                "atr": row["atr"],
                "stoch": row["stoch"],
                "confidence_score": confidence
            }
            
            # Vérifier les conditions de vente
            if symbol in self.positions:
                should_sell, reason = self.check_sell_conditions(
                    symbol, price, indicators, idx
                )
                if should_sell:
                    self.sell(symbol, price, idx, reason)
            
            # Vérifier les conditions d'achat
            else:
                buy_signals = []
                
                if row["rsi"] < 35:
                    buy_signals.append(True)
                if row["macd_hist"] > 0:
                    buy_signals.append(True)
                if price <= row["bb_lower"] * 1.02:
                    buy_signals.append(True)
                if row["stoch"] < 25:
                    buy_signals.append(True)
                
                if len(buy_signals) >= 2 and confidence >= self.min_confidence:
                    self.buy(symbol, price, idx, indicators, risk_score)
            
            # Enregistrer l'équité
            equity = self.cash + sum(
                pos["qty"] * price for pos in self.positions.values()
            )
            self.equity_curve.append((idx, equity))
        
        # Fermer les positions ouvertes
        final_price = df.iloc[-1]["close"]
        for symbol in list(self.positions.keys()):
            self.sell(symbol, final_price, df.index[-1], "Fin du backtest")
        
        # Calculer les métriques
        return self.calculate_metrics()
    
    def calculate_risk_score(self, row: pd.Series, current_equity: float) -> float:
        """Calcule le score de risque simplifié."""
        risk_score = 0.0
        
        # Volatilité ATR
        atr_mean = row.get("atr", 0)
        if atr_mean > 0:
            risk_score += 15
        
        # Confiance inversée
        confidence = calculate_confidence_score_backtest(row)
        risk_score += (100 - confidence) * 0.3
        
        # RSI extrême
        if row["rsi"] < 20 or row["rsi"] > 80:
            risk_score += 15
        
        # Drawdown
        dd_pct = ((self.starting_cash - current_equity) / self.starting_cash) * 100
        if dd_pct > 10:
            risk_score += 15
        elif dd_pct > 5:
            risk_score += 10
        
        return min(100, max(0, risk_score))
    
    def calculate_metrics(self) -> dict:
        """Calcule les métriques de performance du backtest."""
        if not self.trades:
            return {"error": "Aucun trade exécuté"}
        
        # Filtrer les trades de vente
        sell_trades = [t for t in self.trades if t["side"] == "SELL" and "pnl_net" in t]
        
        if not sell_trades:
            return {"error": "Aucun trade de vente"}
        
        # Métriques de base
        total_trades = len(sell_trades)
        winning_trades = [t for t in sell_trades if t["pnl_net"] > 0]
        losing_trades = [t for t in sell_trades if t["pnl_net"] <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = sum(t["pnl_net"] for t in winning_trades)
        total_loss = abs(sum(t["pnl_net"] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Return total
        final_equity = self.cash
        total_return = ((final_equity - self.starting_cash) / self.starting_cash) * 100
        
        # Max drawdown
        equity_curve = pd.Series([eq for _, eq in self.equity_curve])
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (simplifié)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Moyenne des gains/pertes
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "final_equity": final_equity,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_profit": total_profit,
            "total_loss": total_loss
        }
    
    def print_report(self, metrics: dict):
        """Affiche un rapport détaillé des résultats."""
        print(f"\n{'='*80}")
        print(f"RÉSULTATS DU BACKTEST")
        print(f"{'='*80}\n")
        
        if "error" in metrics:
            print(f"❌ {metrics['error']}")
            return
        
        print(f"📊 Statistiques de trading:")
        print(f"   Total trades: {metrics['total_trades']}")
        print(f"   Trades gagnants: {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
        print(f"   Trades perdants: {metrics['losing_trades']}")
        print(f"   Win rate: {metrics['win_rate']:.1f}%")
        print(f"\n💰 Performance:")
        print(f"   Capital initial: {self.starting_cash:.2f}€")
        print(f"   Capital final: {metrics['final_equity']:.2f}€")
        print(f"   Return total: {metrics['total_return']:+.2f}%")
        print(f"   Profit total: {metrics['total_profit']:+.2f}€")
        print(f"   Perte totale: {metrics['total_loss']:.2f}€")
        print(f"\n📈 Métriques avancées:")
        print(f"   Profit factor: {metrics['profit_factor']:.2f}")
        print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Gain moyen: {metrics['avg_win']:+.2f}€")
        print(f"   Perte moyenne: {metrics['avg_loss']:.2f}€")
        print(f"\n{'='*80}\n")

def main():
    """Fonction principale pour exécuter un backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtesting Engine")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Symbole à tester")
    parser.add_argument("--days", type=int, default=30, help="Nombre de jours d'historique")
    parser.add_argument("--cash", type=float, default=100.0, help="Capital initial")
    parser.add_argument("--csv", type=str, help="Fichier CSV avec données historiques")
    parser.add_argument("--fast", action="store_true", help="Mode rapide (indicateurs courts)")
    
    args = parser.parse_args()
    
    # Charger les données
    if args.csv:
        df = load_historical_csv(args.csv)
    else:
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            print("❌ FINNHUB_API_KEY non définie")
            return
        df = fetch_historical_data(args.symbol, api_key, args.days)
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    # Créer le moteur de backtest
    engine = BacktestEngine(starting_cash=args.cash)
    
    # Exécuter le backtest
    metrics = engine.run(df, args.symbol, fast_mode=args.fast)
    
    # Afficher le rapport
    engine.print_report(metrics)

if __name__ == "__main__":
    main()
