# Démarrage Rapide

## Installation sur Raspberry Pi 3B

### 1. Prérequis
\`\`\`bash
# Mettre à jour le système
sudo apt update && sudo apt upgrade -y

# Installer Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Installer Docker Compose
sudo apt install -y docker-compose

# Redémarrer pour appliquer les changements
sudo reboot
\`\`\`

### 2. Configuration
\`\`\`bash
# Cloner ou copier le projet
cd ~/trading-bot

# Copier et configurer .env
cp .env.example .env
nano .env

# Configurer au minimum :
# - FINNHUB_KEYS=votre_clé_api
# - TICKERS=AAPL,TSLA,NVDA
# - STARTING_CASH=100
\`\`\`

### 3. Lancement
\`\`\`bash
# Démarrer tous les services
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Accéder au dashboard
# http://raspberry-pi-ip:8000
\`\`\`

### 4. Vérification
\`\`\`bash
# Vérifier que tout fonctionne
docker-compose ps

# Devrait afficher :
# trading-bot       running
# trading-dashboard running
# trading-db        running
\`\`\`

## Configuration Recommandée pour RPi 3B

\`\`\`env
# .env optimisé pour Raspberry Pi 3B
FINNHUB_KEYS=votre_clé_1,votre_clé_2
TICKERS=AAPL,TSLA
STARTING_CASH=100
FAST_MODE=true
SLEEP_BETWEEN_CALLS_SECONDS=1.0
TICKER_MIN_REFRESH_SECONDS=60
STOP_LOSS_PCT=4.0
TAKE_PROFIT_PCT=3.0
TRAILING_STOP_PCT=1.5
MAX_DRAWDOWN_PCT=15.0
TRADING_PLATFORM=alpaca
\`\`\`

## Commandes Essentielles

\`\`\`bash
# Démarrer
docker-compose up -d

# Arrêter
docker-compose down

# Redémarrer
docker-compose restart

# Voir les logs
docker-compose logs -f trading-bot

# Accéder au dashboard
firefox http://localhost:8000

# Vente d'urgence
touch /tmp/emergency_sell
# ou
docker-compose exec trading-bot touch /tmp/emergency_sell
\`\`\`

## Surveillance

\`\`\`bash
# Ressources système
docker stats

# Température RPi
vcgencmd measure_temp

# Logs en temps réel
docker-compose logs -f

# État des services
docker-compose ps
\`\`\`

## Mise à jour

\`\`\`bash
# Arrêter les services
docker-compose down

# Mettre à jour le code
git pull  # ou copier les nouveaux fichiers

# Reconstruire les images
docker-compose build

# Redémarrer
docker-compose up -d
\`\`\`

## Sauvegarde

\`\`\`bash
# Sauvegarder la base de données
docker-compose exec trading-db pg_dump -U bot trading > backup_$(date +%Y%m%d).sql

# Sauvegarder la configuration
cp .env .env.backup
\`\`\`

## Problèmes Courants

Voir [TROUBLESHOOTING.md](TROUBLESHOOTING.md) pour les solutions détaillées.
