#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Analysis - Optimisé pour Raspberry Pi 3B
Analyse plusieurs échelles de temps pour confirmer les signaux de trading.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class Trend(Enum):
    """Enumération des tendances possibles."""
    BULLISH = "bullish"      # Haussière
    BEARISH = "bearish"      # Baissière
    NEUTRAL = "neutral"      # Neutre
    UNKNOWN = "unknown"      # Inconnu

class TimeframeAnalyzer:
    """
    Analyseur multi-timeframe optimisé pour le Raspberry Pi 3B.
    Utilise des calculs vectorisés pour minimiser l'utilisation CPU/RAM.
    """
    
    # Timeframes supportés (en minutes)
    TIMEFRAMES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }
    
    def __init__(self, fast_mode: bool = False):
        """
        Args:
            fast_mode: Si True, utilise des périodes plus courtes pour les indicateurs
        """
        self.fast_mode = fast_mode
        
        # Périodes des indicateurs selon le mode
        if fast_mode:
            self.sma_short = 5
            self.sma_long = 20
            self.ema_period = 10
        else:
            self.sma_short = 10
            self.sma_long = 50
            self.ema_period = 20
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Rééchantillonne les données 1min vers un timeframe supérieur.
        
        Args:
            df: DataFrame avec index datetime et colonnes OHLCV
            timeframe: Timeframe cible (5m, 15m, 1h, 4h, 1d)
        
        Returns:
            DataFrame rééchantillonné
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Timeframe invalide: {timeframe}")
        
        minutes = self.TIMEFRAMES[timeframe]
        rule = f"{minutes}min" if minutes < 1440 else "1D"
        
        # Rééchantillonnage OHLCV
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df.columns else "first"
        }).dropna()
        
        return resampled
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs de tendance sur un timeframe.
        Optimisé pour minimiser l'utilisation mémoire.
        """
        # SMA
        df["sma_short"] = df["close"].rolling(self.sma_short, min_periods=self.sma_short).mean()
        df["sma_long"] = df["sma_long"] = df["close"].rolling(self.sma_long, min_periods=self.sma_long).mean()
        
        # EMA
        df["ema"] = df["close"].ewm(span=self.ema_period, adjust=False).mean()
        
        # Pente de la SMA longue (tendance)
        df["sma_slope"] = df["sma_long"].diff(5)
        
        return df
    
    def detect_trend(self, df: pd.DataFrame) -> Tuple[Trend, float]:
        """
        Détecte la tendance actuelle sur un timeframe.
        
        Returns:
            (Trend, strength) où strength est entre 0 (faible) et 1 (forte)
        """
        if len(df) < self.sma_long:
            return Trend.UNKNOWN, 0.0
        
        last_row = df.iloc[-1]
        
        # Vérifier si les indicateurs sont disponibles
        if pd.isna(last_row["sma_short"]) or pd.isna(last_row["sma_long"]):
            return Trend.UNKNOWN, 0.0
        
        close = last_row["close"]
        sma_short = last_row["sma_short"]
        sma_long = last_row["sma_long"]
        ema = last_row["ema"]
        sma_slope = last_row["sma_slope"]
        
        # Compteur de signaux haussiers/baissiers
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Signal 1: Prix vs SMA courte
        if close > sma_short:
            bullish_signals += 1
        elif close < sma_short:
            bearish_signals += 1
        total_signals += 1
        
        # Signal 2: SMA courte vs SMA longue
        if sma_short > sma_long:
            bullish_signals += 1
        elif sma_short < sma_long:
            bearish_signals += 1
        total_signals += 1
        
        # Signal 3: Prix vs EMA
        if close > ema:
            bullish_signals += 1
        elif close < ema:
            bearish_signals += 1
        total_signals += 1
        
        # Signal 4: Pente de la SMA longue
        if not pd.isna(sma_slope):
            if sma_slope > 0:
                bullish_signals += 1
            elif sma_slope < 0:
                bearish_signals += 1
            total_signals += 1
        
        # Déterminer la tendance
        if bullish_signals >= 3:
            trend = Trend.BULLISH
            strength = bullish_signals / total_signals
        elif bearish_signals >= 3:
            trend = Trend.BEARISH
            strength = bearish_signals / total_signals
        else:
            trend = Trend.NEUTRAL
            strength = 0.5
        
        return trend, strength
    
    def analyze_multiple_timeframes(self, df_1m: pd.DataFrame, 
                                   timeframes: List[str] = None) -> Dict[str, dict]:
        """
        Analyse plusieurs timeframes et retourne les tendances détectées.
        
        Args:
            df_1m: DataFrame avec données 1min (colonnes: open, high, low, close)
            timeframes: Liste des timeframes à analyser (défaut: 5m, 15m, 1h, 4h)
        
        Returns:
            Dict avec les résultats par timeframe
        """
        if timeframes is None:
            timeframes = ["5m", "15m", "1h", "4h"]
        
        results = {}
        
        for tf in timeframes:
            try:
                # Rééchantillonner
                df_tf = self.resample_data(df_1m, tf)
                
                # Calculer les indicateurs
                df_tf = self.calculate_trend_indicators(df_tf)
                
                # Détecter la tendance
                trend, strength = self.detect_trend(df_tf)
                
                results[tf] = {
                    "trend": trend,
                    "strength": strength,
                    "last_close": float(df_tf.iloc[-1]["close"]) if len(df_tf) > 0 else None,
                    "sma_short": float(df_tf.iloc[-1]["sma_short"]) if not pd.isna(df_tf.iloc[-1]["sma_short"]) else None,
                    "sma_long": float(df_tf.iloc[-1]["sma_long"]) if not pd.isna(df_tf.iloc[-1]["sma_long"]) else None
                }
                
            except Exception as e:
                results[tf] = {
                    "trend": Trend.UNKNOWN,
                    "strength": 0.0,
                    "error": str(e)
                }
        
        return results
    
    def calculate_mtf_score(self, mtf_results: Dict[str, dict]) -> float:
        """
        Calcule un score multi-timeframe (0-100).
        
        Score élevé = toutes les timeframes sont haussières (bon pour acheter)
        Score faible = toutes les timeframes sont baissières (éviter d'acheter)
        
        Args:
            mtf_results: Résultats de analyze_multiple_timeframes
        
        Returns:
            Score entre 0 et 100
        """
        if not mtf_results:
            return 50.0
        
        # Pondération par timeframe (les timeframes longues ont plus de poids)
        weights = {
            "1m": 0.5,
            "5m": 1.0,
            "15m": 1.5,
            "1h": 2.0,
            "4h": 2.5,
            "1d": 3.0
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for tf, result in mtf_results.items():
            trend = result["trend"]
            strength = result["strength"]
            weight = weights.get(tf, 1.0)
            
            if trend == Trend.UNKNOWN:
                continue
            
            # Convertir la tendance en score
            if trend == Trend.BULLISH:
                tf_score = 50 + (strength * 50)  # 50-100
            elif trend == Trend.BEARISH:
                tf_score = 50 - (strength * 50)  # 0-50
            else:  # NEUTRAL
                tf_score = 50
            
            total_score += tf_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 50.0
        
        return total_score / total_weight
    
    def get_mtf_signal(self, mtf_score: float, min_score: float = 60.0) -> Tuple[str, str]:
        """
        Génère un signal de trading basé sur le score multi-timeframe.
        
        Args:
            mtf_score: Score multi-timeframe (0-100)
            min_score: Score minimum pour un signal d'achat
        
        Returns:
            (signal, reason) où signal est "BUY", "SELL", ou "HOLD"
        """
        if mtf_score >= min_score:
            return "BUY", f"MTF score élevé ({mtf_score:.1f}/100) - tendances alignées à la hausse"
        elif mtf_score <= (100 - min_score):
            return "SELL", f"MTF score faible ({mtf_score:.1f}/100) - tendances alignées à la baisse"
        else:
            return "HOLD", f"MTF score neutre ({mtf_score:.1f}/100) - tendances mixtes"
    
    def print_analysis(self, mtf_results: Dict[str, dict], mtf_score: float):
        """Affiche une analyse détaillée des timeframes."""
        print(f"\n{'='*80}")
        print(f"ANALYSE MULTI-TIMEFRAME")
        print(f"{'='*80}\n")
        
        for tf, result in sorted(mtf_results.items(), key=lambda x: self.TIMEFRAMES.get(x[0], 0)):
            trend = result["trend"]
            strength = result["strength"]
            
            # Emoji selon la tendance
            if trend == Trend.BULLISH:
                emoji = "↗"
                color = "HAUSSIER"
            elif trend == Trend.BEARISH:
                emoji = "↘"
                color = "BAISSIER"
            elif trend == Trend.NEUTRAL:
                emoji = "→"
                color = "NEUTRE"
            else:
                emoji = "?"
                color = "INCONNU"
            
            print(f"{tf:>4} {emoji} {color:>10} (force: {strength*100:.0f}%)")
            
            if result.get("sma_short") and result.get("sma_long"):
                print(f"     SMA courte: {result['sma_short']:.2f} | SMA longue: {result['sma_long']:.2f}")
        
        print(f"\n{'─'*80}")
        print(f"SCORE MULTI-TIMEFRAME: {mtf_score:.1f}/100")
        
        signal, reason = self.get_mtf_signal(mtf_score)
        print(f"SIGNAL: {signal} - {reason}")
        print(f"{'='*80}\n")

def integrate_mtf_with_bot(df_ticks: pd.DataFrame, fast_mode: bool = False) -> Tuple[float, Dict]:
    """
    Fonction d'intégration pour le bot de trading principal.
    
    Args:
        df_ticks: DataFrame avec les ticks 1min (colonnes: ts, price)
        fast_mode: Mode rapide pour Raspberry Pi
    
    Returns:
        (mtf_score, mtf_results)
    """
    # Préparer les données OHLC depuis les ticks
    df_1m = df_ticks.set_index("ts") if "ts" in df_ticks.columns else df_ticks
    
    # Créer des colonnes OHLC si elles n'existent pas
    if "open" not in df_1m.columns:
        df_1m = df_1m.resample("1min").agg({
            "price": ["first", "max", "min", "last"]
        })
        df_1m.columns = ["open", "high", "low", "close"]
        df_1m = df_1m.dropna()
    
    # Analyser
    analyzer = TimeframeAnalyzer(fast_mode=fast_mode)
    
    # Utiliser moins de timeframes en mode rapide
    timeframes = ["5m", "15m", "1h"] if fast_mode else ["5m", "15m", "1h", "4h"]
    
    mtf_results = analyzer.analyze_multiple_timeframes(df_1m, timeframes)
    mtf_score = analyzer.calculate_mtf_score(mtf_results)
    
    return mtf_score, mtf_results

def main():
    """Fonction de test pour le module multi-timeframe."""
    import sys
    import os
    
    # Exemple avec des données synthétiques
    print("Test du module Multi-Timeframe Analysis\n")
    
    # Générer des données de test (tendance haussière)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5 + 0.05)  # Tendance haussière
    
    df = pd.DataFrame({
        "open": prices + np.random.randn(1000) * 0.1,
        "high": prices + np.abs(np.random.randn(1000) * 0.2),
        "low": prices - np.abs(np.random.randn(1000) * 0.2),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Analyser
    analyzer = TimeframeAnalyzer(fast_mode=False)
    mtf_results = analyzer.analyze_multiple_timeframes(df, ["5m", "15m", "1h", "4h"])
    mtf_score = analyzer.calculate_mtf_score(mtf_results)
    
    # Afficher les résultats
    analyzer.print_analysis(mtf_results, mtf_score)

if __name__ == "__main__":
    main()
