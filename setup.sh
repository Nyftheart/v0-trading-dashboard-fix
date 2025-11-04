#!/bin/bash

# Trading Bot Setup Script for Raspberry Pi 3B
# This script sets up the trading bot with Docker

set -e

echo "🚀 Trading Bot Setup for Raspberry Pi 3B"
echo "=========================================="

# Check if running on ARM architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "armv7l" && "$ARCH" != "aarch64" ]]; then
    echo "⚠️  Warning: This script is optimized for Raspberry Pi 3B (ARM architecture)"
    echo "   Detected architecture: $ARCH"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed successfully"
    echo "⚠️  Please log out and log back in for group changes to take effect"
    exit 0
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    echo "Installing Docker Compose..."
    sudo apt-get update
    sudo apt-get install -y docker-compose
    echo "✅ Docker Compose installed successfully"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and configuration"
    echo "   nano .env"
    exit 0
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs

# Build and start containers
echo "🐳 Building Docker containers (this may take a while on RPi 3B)..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Dashboard: http://localhost:${WEB_PORT:-8080}"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f bot        # View bot logs"
echo "  docker-compose logs -f dashboard  # View dashboard logs"
echo "  docker-compose ps                 # Check service status"
echo "  docker-compose down               # Stop all services"
echo "  docker-compose restart bot        # Restart bot"
echo ""
