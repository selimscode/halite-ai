# Halite AI Agent | Top 2% of 6,000+

Autonomous agent for [Two Sigma's Halite Competition](https://www.kaggle.com/competitions/halite) on Kaggle. Ships collect halite (resource) from a grid-based ocean, deposit at shipyards, and compete against opponents. The agent ranked in the **top 2% globally** against 6,000+ competitors.

## Approach

Rather than pure reinforcement learning, this agent uses **multi-step lookahead optimization** with strategic heuristics:

### 1. Collection Optimization
4-move lookahead that evaluates all possible movement sequences, scoring each path by expected halite yield:

```
For each ship:
  For each 4-move sequence (5^4 = 625 combinations):
    Calculate cumulative halite from staying/collecting
    Avoid collisions with allies and shipyards
    Select highest-yield path
```

### 2. Shipyard Density Analysis
Smart ship spawning based on local resource availability:
- Calculate halite-to-ship ratio around each shipyard
- Spawn new ships at shipyards with highest resource density
- Prevents overcrowding in depleted areas

### 3. Dynamic Expansion
Automatic base-building when ships venture far from home:
- If nearest shipyard > 25% of board size away, convert ship to shipyard
- Maintains efficient deposit routes across the map

### 4. Combat Tactics
Adaptive attack/escape based on relative cargo:
- **Attack**: Chase enemies carrying 2x more halite (profitable collision)
- **Escape**: Flee from threats when carrying valuable cargo
- Manhattan distance tracking for all nearby enemies

## Architecture

```
agent.py
├── agent()                    # Main entry point, called each turn
├── ship_dispatcher()          # Collection/deposit orchestration
├── shipyard_halite_density()  # Resource density analysis
├── nearest_shipyard()         # Pathfinding helper
└── move_ship()                # Direction calculation
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `HALITE_THRESHOLD` | 500 | Trigger deposit run |
| `HALITE_TO_SHIP_THRESHOLD` | 1400 | Spawn new ships when ratio exceeds |
| `ENEMY_DIST_THRESHOLD` | 2 | Attack/escape trigger distance |
| `FACTOR_NEAREST_SHIPYARD` | 0.25 | Convert to shipyard if farther |

## Results

- **Final Score**: 671.4
- **Ranking**: Top 2% of 6,000+ participants
- **Competition**: Halite by Two Sigma (Kaggle, 2020)

## Run Locally

```bash
pip install kaggle-environments

# Test against random agent
python -c "
from kaggle_environments import make
from agent import agent
env = make('halite', debug=True)
env.run([agent, 'random'])
env.render(mode='ipython')
"
```

## What I Learned

- Lookahead search beats reactive strategies in resource games
- Local density metrics enable smarter macro decisions
- Simple attack/escape heuristics provide significant edge
- Collision avoidance is critical in multi-agent environments
