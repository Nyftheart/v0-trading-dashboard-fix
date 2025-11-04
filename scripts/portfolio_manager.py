#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Manager & Volume Analysis - Optimisé pour Raspberry Pi 3B
Gestion avancée du portefeuille et analyse du volume de trading.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    """Représente une position dans le portefeuille."""
    symbol: str
    qty: float
    entry_price: float
    current_price: float
    entry_time: datetime
    entry_fees: float
    
    @property
    def value(self) -> float:
        """Valeur actuelle de la position."""
        return self.qty * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Coût total d'acquisition."""
        return (self.qty * self.entry_price) + self.entry_fees
    
    @property
    def pnl(self) -> float:
        """Profit/Loss non réalisé."""
        return self.value - self.cost_basis
    
    @property
    def pnl_pct(self) -> float:
        """Profit/Loss en pourcentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl / self.cost_basis) * 100
    
    @property
    def weight(self) -> float:
        """Poids dans le portefeuille (sera calculé par le PortfolioManager)."""
        return 0.0

class VolumeAnalyzer:
    """
    Analyseur de volume optimisé pour le Raspberry Pi 3B.
    Détecte les breakouts, accumulations et divergences.
    """
    
    def __init__(self, lookback_periods: int = 20):
        """
        Args:
            lookback_periods: Nombre de périodes pour les calculs de moyenne
        """
        self.lookback_periods = lookback_periods
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule l'On-Balance Volume (OBV).
        OBV monte quand le prix monte avec du volume, descend sinon.
        """
        if "volume" not in df.columns or "close" not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        price_change = df["close"].diff()
        obv = (np.sign(price_change) * df["volume"]).fillna(0).cumsum()
        return obv
    
    def calculate_volume_sma(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calcule la moyenne mobile du volume."""
        if period is None:
            period = self.lookback_periods
        
        if "volume" not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        return df["volume"].rolling(period, min_periods=period).mean()
    
    def detect_volume_breakout(self, df: pd.DataFrame, threshold: float = 2.0) -> Tuple[bool, float]:
        """
        Détecte un breakout de volume.
        
        Args:
            df: DataFrame avec colonnes volume et close
            threshold: Multiplicateur du volume moyen pour détecter un breakout
        
        Returns:
            (is_breakout, volume_ratio)
        """
        if len(df) < self.lookback_periods or "volume" not in df.columns:
            return False, 0.0
        
        current_volume = df.iloc[-1]["volume"]
        avg_volume = df["volume"].tail(self.lookback_periods).mean()
        
        if avg_volume == 0:
            return False, 0.0
        
        volume_ratio = current_volume / avg_volume
        is_breakout = volume_ratio >= threshold
        
        return is_breakout, volume_ratio
    
    def detect_accumulation_distribution(self, df: pd.DataFrame) -> str:
        """
        Détecte si on est en phase d'accumulation ou de distribution.
        
        Returns:
            "accumulation", "distribution", ou "neutral"
        """
        if len(df) < self.lookback_periods:
            return "neutral"
        
        obv = self.calculate_obv(df)
        
        if len(obv) < self.lookback_periods:
            return "neutral"
        
        # Tendance de l'OBV
        obv_recent = obv.tail(self.lookback_periods)
        obv_slope = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / self.lookback_periods
        
        # Tendance du prix
        price_recent = df["close"].tail(self.lookback_periods)
        price_slope = (price_recent.iloc[-1] - price_recent.iloc[0]) / self.lookback_periods
        
        # Accumulation: OBV monte, prix stable ou monte
        if obv_slope > 0 and price_slope >= 0:
            return "accumulation"
        
        # Distribution: OBV descend, prix stable ou descend
        elif obv_slope < 0 and price_slope <= 0:
            return "distribution"
        
        return "neutral"
    
    def detect_price_volume_divergence(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Détecte les divergences entre prix et volume.
        
        Returns:
            (has_divergence, divergence_type)
            divergence_type: "bullish" (prix baisse, OBV monte) ou "bearish" (prix monte, OBV baisse)
        """
        if len(df) < self.lookback_periods:
            return False, ""
        
        obv = self.calculate_obv(df)
        
        if len(obv) < self.lookback_periods:
            return False, ""
        
        # Tendances récentes
        obv_recent = obv.tail(self.lookback_periods)
        price_recent = df["close"].tail(self.lookback_periods)
        
        obv_trend = obv_recent.iloc[-1] - obv_recent.iloc[0]
        price_trend = price_recent.iloc[-1] - price_recent.iloc[0]
        
        # Divergence haussière: prix baisse mais OBV monte
        if price_trend < 0 and obv_trend > 0:
            return True, "bullish"
        
        # Divergence baissière: prix monte mais OBV baisse
        elif price_trend > 0 and obv_trend < 0:
            return True, "bearish"
        
        return False, ""
    
    def calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        Calcule un score de volume 0-100.
        Score élevé = conditions de volume favorables pour un achat.
        """
        if len(df) < self.lookback_periods or "volume" not in df.columns:
            return 50.0
        
        score = 0.0
        factors = 0
        
        # Facteur 1: Volume breakout (30 points)
        is_breakout, volume_ratio = self.detect_volume_breakout(df)
        if is_breakout:
            score += 30
        elif volume_ratio > 1.5:
            score += 15
        factors += 1
        
        # Facteur 2: Accumulation/Distribution (30 points)
        phase = self.detect_accumulation_distribution(df)
        if phase == "accumulation":
            score += 30
        elif phase == "neutral":
            score += 15
        factors += 1
        
        # Facteur 3: Divergence prix/volume (40 points)
        has_div, div_type = self.detect_price_volume_divergence(df)
        if has_div and div_type == "bullish":
            score += 40
        elif not has_div:
            score += 20
        factors += 1
        
        if factors == 0:
            return 50.0
        
        return (score / factors) * (100 / 33.33)  # Normaliser à 100

class PortfolioManager:
    """
    Gestionnaire de portefeuille optimisé pour le Raspberry Pi 3B.
    Gère la diversification, les limites et le rebalancing.
    """
    
    def __init__(self, max_positions: int = 5, max_position_pct: float = 30.0,
                 min_position_pct: float = 10.0, rebalance_threshold: float = 10.0):
        """
        Args:
            max_positions: Nombre maximum de positions simultanées
            max_position_pct: Poids maximum d'une position (% du portefeuille)
            min_position_pct: Poids minimum d'une position (% du portefeuille)
            rebalance_threshold: Seuil de déséquilibre pour déclencher un rebalancing (%)
        """
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.rebalance_threshold = rebalance_threshold
        
        self.positions: Dict[str, Position] = {}
        self.cash = 0.0
        self.total_equity = 0.0
    
    def update_positions(self, positions_data: List[dict], cash: float):
        """
        Met à jour les positions du portefeuille.
        
        Args:
            positions_data: Liste de dicts avec symbol, qty, entry, current_price, etc.
            cash: Cash disponible
        """
        self.cash = cash
        self.positions = {}
        
        for pos_data in positions_data:
            pos = Position(
                symbol=pos_data["symbol"],
                qty=pos_data["qty"],
                entry_price=pos_data["entry"],
                current_price=pos_data["current_price"],
                entry_time=pos_data.get("entry_time", datetime.now()),
                entry_fees=pos_data.get("entry_fees", 0.0)
            )
            self.positions[pos.symbol] = pos
        
        self.total_equity = self.cash + sum(pos.value for pos in self.positions.values())
    
    def get_position_weight(self, symbol: str) -> float:
        """Retourne le poids d'une position dans le portefeuille (%)."""
        if symbol not in self.positions or self.total_equity == 0:
            return 0.0
        
        return (self.positions[symbol].value / self.total_equity) * 100
    
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """
        Vérifie si on peut ouvrir une nouvelle position.
        
        Returns:
            (can_open, reason)
        """
        # Vérifier le nombre maximum de positions
        if len(self.positions) >= self.max_positions:
            return False, f"Nombre maximum de positions atteint ({self.max_positions})"
        
        # Vérifier si on a déjà une position sur ce symbole
        if symbol in self.positions:
            return False, f"Position déjà ouverte sur {symbol}"
        
        return True, ""
    
    def calculate_optimal_position_size(self, symbol: str, proposed_size: float) -> Tuple[float, str]:
        """
        Calcule la taille optimale d'une position en tenant compte des limites.
        
        Args:
            symbol: Symbole de l'action
            proposed_size: Taille proposée en €
        
        Returns:
            (optimal_size, reason)
        """
        if self.total_equity == 0:
            return proposed_size, "Équité nulle"
        
        # Calculer le poids proposé
        proposed_weight = (proposed_size / self.total_equity) * 100
        
        # Vérifier le poids maximum
        if proposed_weight > self.max_position_pct:
            optimal_size = (self.max_position_pct / 100) * self.total_equity
            return optimal_size, f"Limité à {self.max_position_pct}% du portefeuille"
        
        # Vérifier le poids minimum
        if proposed_weight < self.min_position_pct:
            optimal_size = (self.min_position_pct / 100) * self.total_equity
            if optimal_size <= self.cash:
                return optimal_size, f"Augmenté à {self.min_position_pct}% minimum"
            else:
                return proposed_size, "Cash insuffisant pour atteindre le minimum"
        
        return proposed_size, "Taille optimale"
    
    def needs_rebalancing(self) -> Tuple[bool, List[str]]:
        """
        Vérifie si le portefeuille nécessite un rebalancing.
        
        Returns:
            (needs_rebalancing, reasons)
        """
        if not self.positions or self.total_equity == 0:
            return False, []
        
        reasons = []
        
        # Vérifier les positions trop grandes
        for symbol, pos in self.positions.items():
            weight = self.get_position_weight(symbol)
            if weight > self.max_position_pct + self.rebalance_threshold:
                reasons.append(f"{symbol} trop grand ({weight:.1f}% > {self.max_position_pct}%)")
        
        # Vérifier les positions trop petites
        for symbol, pos in self.positions.items():
            weight = self.get_position_weight(symbol)
            if weight < self.min_position_pct - self.rebalance_threshold:
                reasons.append(f"{symbol} trop petit ({weight:.1f}% < {self.min_position_pct}%)")
        
        return len(reasons) > 0, reasons
    
    def get_portfolio_stats(self) -> dict:
        """Retourne les statistiques du portefeuille."""
        if not self.positions:
            return {
                "total_equity": self.total_equity,
                "cash": self.cash,
                "num_positions": 0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "positions": []
            }
        
        total_pnl = sum(pos.pnl for pos in self.positions.values())
        total_cost = sum(pos.cost_basis for pos in self.positions.values())
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        
        positions_stats = []
        for symbol, pos in self.positions.items():
            positions_stats.append({
                "symbol": symbol,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "value": pos.value,
                "cost_basis": pos.cost_basis,
                "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct,
                "weight": self.get_position_weight(symbol)
            })
        
        return {
            "total_equity": self.total_equity,
            "cash": self.cash,
            "cash_pct": (self.cash / self.total_equity * 100) if self.total_equity > 0 else 0,
            "num_positions": len(self.positions),
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": sorted(positions_stats, key=lambda x: x["weight"], reverse=True)
        }
    
    def print_portfolio_summary(self):
        """Affiche un résumé du portefeuille."""
        stats = self.get_portfolio_stats()
        
        print(f"\n{'='*80}")
        print(f"RÉSUMÉ DU PORTEFEUILLE")
        print(f"{'='*80}\n")
        
        print(f"Équité totale: {stats['total_equity']:.2f}€")
        print(f"Cash disponible: {stats['cash']:.2f}€ ({stats['cash_pct']:.1f}%)")
        print(f"Nombre de positions: {stats['num_positions']}/{self.max_positions}")
        print(f"PnL total: {stats['total_pnl']:+.2f}€ ({stats['total_pnl_pct']:+.2f}%)")
        
        if stats['positions']:
            print(f"\n{'─'*80}")
            print(f"{'Symbole':<10} {'Poids':<10} {'Valeur':<12} {'PnL':<15} {'PnL %':<10}")
            print(f"{'─'*80}")
            
            for pos in stats['positions']:
                print(f"{pos['symbol']:<10} {pos['weight']:>6.1f}%   "
                      f"{pos['value']:>10.2f}€  {pos['pnl']:>+10.2f}€  "
                      f"{pos['pnl_pct']:>+8.2f}%")
        
        print(f"{'='*80}\n")

def integrate_volume_analysis(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, dict]:
    """
    Fonction d'intégration pour l'analyse de volume.
    
    Args:
        df: DataFrame avec colonnes close et volume
        lookback: Nombre de périodes pour les calculs
    
    Returns:
        (volume_score, volume_details)
    """
    analyzer = VolumeAnalyzer(lookback_periods=lookback)
    
    volume_score = analyzer.calculate_volume_score(df)
    
    is_breakout, volume_ratio = analyzer.detect_volume_breakout(df)
    phase = analyzer.detect_accumulation_distribution(df)
    has_div, div_type = analyzer.detect_price_volume_divergence(df)
    
    volume_details = {
        "score": volume_score,
        "is_breakout": is_breakout,
        "volume_ratio": volume_ratio,
        "phase": phase,
        "has_divergence": has_div,
        "divergence_type": div_type if has_div else None
    }
    
    return volume_score, volume_details

def main():
    """Fonction de test pour le module."""
    print("Test du module Portfolio Manager & Volume Analysis\n")
    
    # Test Volume Analysis
    print("Test Volume Analysis:")
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    volumes = np.random.randint(1000, 10000, 100)
    
    # Simuler un breakout de volume
    volumes[-1] = 25000
    
    df = pd.DataFrame({
        "close": prices,
        "volume": volumes
    }, index=dates)
    
    volume_score, volume_details = integrate_volume_analysis(df)
    
    print(f"Score de volume: {volume_score:.1f}/100")
    print(f"Breakout: {volume_details['is_breakout']} (ratio: {volume_details['volume_ratio']:.2f}x)")
    print(f"Phase: {volume_details['phase']}")
    print(f"Divergence: {volume_details['has_divergence']} ({volume_details['divergence_type']})")
    
    # Test Portfolio Manager
    print(f"\n{'='*80}\n")
    print("Test Portfolio Manager:")
    
    pm = PortfolioManager(max_positions=5, max_position_pct=30.0)
    
    # Simuler des positions
    positions_data = [
        {"symbol": "AAPL", "qty": 10, "entry": 150.0, "current_price": 155.0, "entry_fees": 0.5},
        {"symbol": "TSLA", "qty": 5, "entry": 200.0, "current_price": 210.0, "entry_fees": 0.5},
        {"symbol": "NVDA", "qty": 8, "entry": 300.0, "current_price": 320.0, "entry_fees": 0.5}
    ]
    
    pm.update_positions(positions_data, cash=2000.0)
    pm.print_portfolio_summary()
    
    # Vérifier si on peut ouvrir une nouvelle position
    can_open, reason = pm.can_open_position("GOOGL")
    print(f"Peut ouvrir GOOGL: {can_open} - {reason}")
    
    # Vérifier le rebalancing
    needs_rebal, reasons = pm.needs_rebalancing()
    print(f"Besoin de rebalancing: {needs_rebal}")
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")

if __name__ == "__main__":
    main()
