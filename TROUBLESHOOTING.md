# Guide de Dépannage

## Erreurs Courantes

### 1. Erreur: "column 'qty' does not exist"

**Cause:** Le schéma de la base de données ne correspond pas au code du bot.

**Solution:**
\`\`\`bash
# Recréer la base de données avec le bon schéma
docker-compose down -v
docker-compose up -d trading-db
sleep 5
docker-compose up -d
\`\`\`

### 2. Erreur: "Directory 'static' does not exist"

**Cause:** Le répertoire static n'existe pas.

**Solution:** Le répertoire a été créé automatiquement. Si l'erreur persiste :
\`\`\`bash
mkdir -p static templates
\`\`\`

### 3. Erreur: "Connection refused" ou "Database not ready"

**Cause:** La base de données PostgreSQL n'est pas encore prête.

**Solution:**
\`\`\`bash
# Attendre que la base de données soit prête
docker-compose logs -f trading-db

# Redémarrer les services
docker-compose restart trading-bot trading-dashboard
\`\`\`

### 4. Erreur: "API key exhausted" ou "429 Too Many Requests"

**Cause:** Limite de requêtes API Finnhub atteinte.

**Solution:**
- Ajouter plus de clés API dans `.env` (FINNHUB_KEYS)
- Augmenter SLEEP_BETWEEN_CALLS_SECONDS
- Réduire le nombre de tickers

### 5. Bot redémarre en boucle

**Cause:** Erreur dans le code ou configuration incorrecte.

**Solution:**
\`\`\`bash
# Voir les logs détaillés
docker-compose logs -f trading-bot

# Vérifier la configuration
cat .env

# Redémarrer proprement
docker-compose down
docker-compose up -d
\`\`\`

## Commandes Utiles

### Voir les logs en temps réel
\`\`\`bash
docker-compose logs -f trading-bot
docker-compose logs -f trading-dashboard
docker-compose logs -f trading-db
\`\`\`

### Redémarrer un service
\`\`\`bash
docker-compose restart trading-bot
docker-compose restart trading-dashboard
\`\`\`

### Accéder à la base de données
\`\`\`bash
docker-compose exec trading-db psql -U bot -d trading

# Commandes SQL utiles
\dt                          # Lister les tables
SELECT * FROM positions;     # Voir les positions
SELECT * FROM trades ORDER BY ts DESC LIMIT 10;  # Derniers trades
SELECT * FROM equity ORDER BY ts DESC LIMIT 10;  # Historique équité
\`\`\`

### Nettoyer complètement
\`\`\`bash
# Arrêter et supprimer tout (y compris les données)
docker-compose down -v

# Reconstruire les images
docker-compose build --no-cache

# Redémarrer
docker-compose up -d
\`\`\`

### Vérifier l'état des services
\`\`\`bash
docker-compose ps
docker-compose top
\`\`\`

### Sauvegarder la base de données
\`\`\`bash
docker-compose exec trading-db pg_dump -U bot trading > backup.sql
\`\`\`

### Restaurer la base de données
\`\`\`bash
cat backup.sql | docker-compose exec -T trading-db psql -U bot trading
\`\`\`

## Optimisation pour Raspberry Pi 3B

### Réduire l'utilisation mémoire
\`\`\`bash
# Dans .env, ajuster :
FAST_MODE=true
TICKERS=AAPL,TSLA  # Réduire le nombre de tickers
SLEEP_BETWEEN_CALLS_SECONDS=1.0  # Augmenter le délai
\`\`\`

### Surveiller les ressources
\`\`\`bash
# CPU et mémoire
docker stats

# Température du Raspberry Pi
vcgencmd measure_temp
\`\`\`

### Limiter la mémoire Docker
\`\`\`bash
# Dans docker-compose.yml, les limites sont déjà configurées :
# Bot: 512MB
# Dashboard: 256MB
# Database: 512MB
\`\`\`

## Support

Si les problèmes persistent :
1. Vérifier les logs : `docker-compose logs -f`
2. Vérifier la configuration : `cat .env`
3. Vérifier l'espace disque : `df -h`
4. Vérifier la mémoire : `free -h`
