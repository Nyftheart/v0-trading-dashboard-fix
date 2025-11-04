# 🤖 Advanced Trading Bot

Bot de trading automatisé avec analyse multi-timeframe, gestion de risque avancée et backtesting. Optimisé pour Raspberry Pi 3B.

## 🚀 Fonctionnalités

### Analyse Technique
- **Multi-timeframe analysis** : Analyse sur 5min, 15min, 1h, 4h
- **Indicateurs avancés** : RSI, MACD, SMA, Bollinger Bands, Stochastic, ATR
- **Volume analysis** : Détection de breakouts et accumulations
- **Score de confiance** : Filtrage intelligent des trades

### Gestion du Risque
- **Position sizing dynamique** : Basé sur la volatilité (ATR)
- **Stop-loss & Take-profit** : Fixe et dynamique
- **Trailing stop** : Sécurise les gains automatiquement
- **Quick profit lock** : Verrouillage rapide des petits gains
- **Safety score** : Optimise les sorties en profit
- **Anti-panic sell** : Garde les positions en survente
- **Max drawdown protection** : Arrêt d'urgence si perte > 15%

### Gestion de Portefeuille
- **Diversification automatique** : Limite par ticker
- **Corrélation** : Évite les positions trop corrélées
- **Rebalancing** : Optimisation automatique

### Outils
- **Backtesting engine** : Test sur données historiques
- **Métriques avancées** : Win rate, Sharpe ratio, profit factor
- **Dashboard web** : Suivi en temps réel
- **Frais réalistes** : Alpaca fees (ultra-faibles)

## 📋 Prérequis

- Raspberry Pi 3B (ou supérieur) avec Raspbian/Debian
- Docker & Docker Compose
- Clé API Finnhub (gratuite sur [finnhub.io](https://finnhub.io))

## 🛠️ Installation

### 1. Cloner le projet

\`\`\`bash
git clone <your-repo-url>
cd trading-bot
\`\`\`

### 2. Configuration

\`\`\`bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer avec vos paramètres
nano .env
\`\`\`

**Configuration minimale requise :**
- `FINNHUB_KEYS` : Votre clé API Finnhub
- `TICKERS` : Liste des actions à trader
- `STARTING_CASH` : Capital initial

### 3. Installation automatique

\`\`\`bash
chmod +x setup.sh
./setup.sh
\`\`\`

Le script va :
- Vérifier Docker et Docker Compose
- Créer les répertoires nécessaires
- Construire les containers
- Démarrer les services

### 4. Installation manuelle (alternative)

\`\`\`bash
# Créer les répertoires
mkdir -p data logs

# Construire les containers
docker-compose build

# Démarrer les services
docker-compose up -d
\`\`\`

## 🎮 Utilisation

### Démarrer le bot

\`\`\`bash
docker-compose up -d
\`\`\`

### Voir les logs

\`\`\`bash
# Logs du bot
docker-compose logs -f bot

# Logs du dashboard
docker-compose logs -f dashboard

# Tous les logs
docker-compose logs -f
\`\`\`

### Accéder au dashboard

Ouvrez votre navigateur : `http://localhost:8080`

### Arrêter le bot

\`\`\`bash
docker-compose down
\`\`\`

### Redémarrer le bot

\`\`\`bash
docker-compose restart bot
\`\`\`

### Vente d'urgence

\`\`\`bash
# Créer le fichier trigger
touch /tmp/emergency_sell

# Ou définir la variable d'environnement
docker-compose exec bot bash -c "export EMERGENCY_SELL=true"
\`\`\`

## 📊 Backtesting

\`\`\`bash
# Lancer un backtest
docker-compose exec bot python scripts/backtesting.py \
  --symbol TSLA \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --capital 1000

# Avec données CSV
docker-compose exec bot python scripts/backtesting.py \
  --csv data/TSLA_historical.csv \
  --capital 1000
\`\`\`

## ⚙️ Configuration Avancée

### Variables d'environnement principales

| Variable | Description | Défaut |
|----------|-------------|--------|
| `STARTING_CASH` | Capital initial | 100 |
| `STOP_LOSS_PCT` | Stop-loss en % | 4.0 |
| `TAKE_PROFIT_PCT` | Take-profit en % | 3.0 |
| `TRAILING_STOP_PCT` | Trailing stop en % | 1.5 |
| `MAX_DRAWDOWN_PCT` | Drawdown max avant arrêt | 15.0 |
| `SAFETY_SCORE_THRESHOLD` | Seuil de vente (score de sûreté) | 60 |
| `QUICK_PROFIT_THRESHOLD` | Gain min pour quick profit | 0.5 |
| `TRADING_PLATFORM` | Plateforme (alpaca/degiro) | alpaca |

### Optimisation pour Raspberry Pi 3B

Le bot est optimisé pour fonctionner avec 1GB de RAM :
- Limites mémoire Docker : 512MB (bot) + 256MB (dashboard)
- Calculs vectorisés avec NumPy
- Cache intelligent des données
- Garbage collection optimisé

## 🐛 Dépannage

### Le bot ne démarre pas

\`\`\`bash
# Vérifier les logs
docker-compose logs bot

# Vérifier la configuration
docker-compose config

# Reconstruire les containers
docker-compose build --no-cache
docker-compose up -d
\`\`\`

### Problèmes de mémoire sur RPi 3B

\`\`\`bash
# Réduire le nombre de tickers dans .env
TICKERS=TSLA,NVDA,AMD,AAPL

# Augmenter le swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
\`\`\`

### Base de données corrompue

\`\`\`bash
# Supprimer et recréer
docker-compose down -v
docker-compose up -d
\`\`\`

## 📈 Performances

Sur Raspberry Pi 3B :
- Consommation mémoire : ~400-600MB
- CPU : 20-40% en moyenne
- Latence API : ~100-300ms
- Capacité : 10-30 tickers simultanés

## 🔒 Sécurité

- Ne jamais commiter le fichier `.env`
- Utiliser des clés API en lecture seule si possible
- Limiter l'accès au dashboard (firewall)
- Sauvegarder régulièrement la base de données

## 📝 Licence

MIT License - Utilisez à vos propres risques

## ⚠️ Avertissement

Ce bot est fourni à titre éducatif. Le trading comporte des risques de perte en capital. Testez toujours en mode simulation avant d'utiliser de l'argent réel.
