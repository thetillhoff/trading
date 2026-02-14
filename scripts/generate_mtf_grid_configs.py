#!/usr/bin/env python3
"""Generate MTF ensemble grid search configs."""
from pathlib import Path

BASE_CONFIG = """name: {name}
description: '{description}'
indicators:
  elliott_wave:
    enabled: true
    min_confidence: 0.65
    min_wave_size: 0.025
  rsi:
    period: 5
    oversold: 25
    overbought: 75
  ema:
    short_period: 20
    long_period: 50
  macd:
    fast: 12
    slow: 26
    signal: 12
risk:
  risk_reward: 3.0
  position_size_pct: 0.1
  max_positions_per_instrument: 2
  min_position_size: 20
  use_confidence_sizing: true
signals:
  signal_types: all
  min_certainty: 0.7
  use_trend_filter: false
  use_multi_timeframe: true
  use_multi_timeframe_filter: false
  indicator_weights:
    rsi: {rsi_weight}
    ema: 0.075
    macd: 0.075
    mtf:
{mtf_config}
regime:
  use_regime_detection: false
  invert_signals_in_bull: true
costs:
  interest_rate_pa: 0.02
  trade_fee_absolute: 0
  trade_fee_pct: 0.0005
  trade_fee_min: 1
  trade_fee_max: 10
evaluation:
  step_days: 1
  lookback_days: 365
  initial_capital: 10000
data:
  start_date: '2008-01-01'
  end_date: '2012-01-01'
"""

def main():
    project_root = Path(__file__).parent.parent
    single_dir = project_root / "configs" / "grid_mtf_ensemble_single"
    dual_dir = project_root / "configs" / "grid_mtf_ensemble_dual"
    
    single_dir.mkdir(parents=True, exist_ok=True)
    dual_dir.mkdir(parents=True, exist_ok=True)
    
    # Single MTF configs: periods 2,4,6,8,10,12 x weights 0.15,0.25,0.35
    periods = [2, 4, 6, 8, 10, 12]
    weights = [0.15, 0.25, 0.35]
    
    print(f"Generating {len(periods) * len(weights)} single MTF configs...")
    for period in periods:
        for weight in weights:
            name = f"mtf_ens_{period}w_{int(weight*100):03d}"
            desc = f"[mtf_ensemble_single] period={period}w weight={weight}"
            mtf_config = f"      - period: {period}\n        weight: {weight}"
            
            content = BASE_CONFIG.format(
                name=name,
                description=desc,
                rsi_weight=0.6,
                mtf_config=mtf_config
            )
            
            filepath = single_dir / f"mtf_{period}w_{int(weight*100):03d}.yaml"
            filepath.write_text(content)
            print(f"  {filepath.name}")
    
    # Dual MTF configs: 4w+8w with various weight combinations and RSI variations
    dual_configs = [
        # (4w_weight, 8w_weight, rsi_weight)
        (0.15, 0.15, 0.40),
        (0.15, 0.15, 0.50),
        (0.15, 0.15, 0.60),
        (0.20, 0.10, 0.40),
        (0.20, 0.10, 0.50),
        (0.20, 0.10, 0.60),
        (0.10, 0.20, 0.40),
        (0.10, 0.20, 0.50),
        (0.10, 0.20, 0.60),
    ]
    
    print(f"\nGenerating {len(dual_configs)} dual MTF configs...")
    for w4, w8, rsi_w in dual_configs:
        name = f"mtf_dual_4w{int(w4*100):03d}_8w{int(w8*100):03d}_rsi{int(rsi_w*100):03d}"
        desc = f"[mtf_ensemble_dual] 4w={w4} 8w={w8} rsi={rsi_w}"
        mtf_config = f"      - period: 4\n        weight: {w4}\n      - period: 8\n        weight: {w8}"
        
        content = BASE_CONFIG.format(
            name=name,
            description=desc,
            rsi_weight=rsi_w,
            mtf_config=mtf_config
        )
        
        filepath = dual_dir / f"dual_4w{int(w4*100):03d}_8w{int(w8*100):03d}_rsi{int(rsi_w*100):03d}.yaml"
        filepath.write_text(content)
        print(f"  {filepath.name}")
    
    print(f"\nDone! Created {len(periods) * len(weights)} single + {len(dual_configs)} dual = {len(periods) * len(weights) + len(dual_configs)} total configs")

if __name__ == "__main__":
    main()
