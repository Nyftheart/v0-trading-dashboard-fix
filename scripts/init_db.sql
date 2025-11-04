-- Initialize trading database schema

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 4) NOT NULL,
    entry_time TIMESTAMP NOT NULL DEFAULT NOW(),
    entry_fees DECIMAL(18, 4) DEFAULT 0,
    highest_price DECIMAL(18, 4),
    stop_loss DECIMAL(18, 4),
    take_profit DECIMAL(18, 4),
    status VARCHAR(20) DEFAULT 'open',
    CONSTRAINT unique_open_position UNIQUE (symbol, status)
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 4) NOT NULL,
    fees DECIMAL(18, 4) DEFAULT 0,
    pnl DECIMAL(18, 4),
    pnl_pct DECIMAL(10, 4),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    reason TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    cash DECIMAL(18, 4) NOT NULL,
    equity DECIMAL(18, 4) NOT NULL,
    total_pnl DECIMAL(18, 4) NOT NULL,
    total_pnl_pct DECIMAL(10, 4) NOT NULL,
    num_positions INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(18, 4) NOT NULL,
    metadata JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);

-- Insert initial portfolio state
INSERT INTO portfolio_history (cash, equity, total_pnl, total_pnl_pct, num_positions)
VALUES (100.0, 100.0, 0.0, 0.0, 0)
ON CONFLICT DO NOTHING;
