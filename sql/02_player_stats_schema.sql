-- ============================================================
-- Player stats schema: box scores pulled via nba_api
-- ============================================================

CREATE TABLE IF NOT EXISTS player_stats (
    game_id         VARCHAR(20) PRIMARY KEY,   -- nba_api GAME_ID
    player_name     VARCHAR(100) NOT NULL DEFAULT 'Stephen Curry',
    game_date       DATE NOT NULL,
    matchup         VARCHAR(20),               -- e.g. 'OKC vs. LAL' or 'OKC @ LAL'
    opponent        VARCHAR(10),               -- team abbreviation
    home_away       VARCHAR(4),                -- 'HOME' | 'AWAY'
    win_loss        VARCHAR(1),                -- 'W' | 'L'
    minutes         NUMERIC(4,1),
    points           INTEGER,
    rebounds        INTEGER,
    assists         INTEGER,
    steals          INTEGER,
    blocks          INTEGER,
    turnovers       INTEGER,
    fg_pct          NUMERIC(4,3),
    fg3_pct         NUMERIC(4,3),
    ft_pct          NUMERIC(4,3),
    plus_minus      NUMERIC(5,1),
    fetched_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_player_stats_game_date ON player_stats(game_date);
