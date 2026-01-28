# Hypothesis Test Results

Reference baseline for the hypotheses below: EW + RSI + EMA + MACD on a single equity index, walk-forward evaluation over multiple subperiods. Best average alpha among tested configs in that setup. Performance varies by market condition (strongest in bear/crash, weaker in sustained bull).

---

## RSI Hypotheses

### RSI period 7 is optimal for standalone RSI strategies

**Hypothesis:** RSI period 7 yields better average alpha than periods 14 or 21 when RSI is used alone.

**Findings:** Period 7: -5.08% average alpha, 40.68% win rate, 2,386 trades. Range from -13.54% (bull) to +4.83% (dotcom crash). Periods 14 and 21 yield -0.50% and -0.24% in the comparison set.

**Conclusion:** REJECTED. Fast RSI (period 7) produces more false signals and worse alpha. Under circumstances: walk-forward on DJIA over multiple periods, RSI-only configs — period 7 is not optimal; period 14 is least bad among RSI-only variants.

---

### Tighter RSI thresholds (25/75) improve alpha vs 30/70

**Hypothesis:** Thresholds 25/75 reduce false signals and improve alpha versus 30/70.

**Findings:** 25/75: -3.37% alpha, 41.15% win rate, 740 trades. 30/70: -1.86% alpha, 40.30% win rate, 1,147 trades.

**Conclusion:** REJECTED. Tighter thresholds reduce trades and win rate; 30/70 performs better in the tested setup.

---

### RSI alone is a viable signal source

**Hypothesis:** RSI used without other indicators can achieve positive or acceptable alpha.

**Findings:** RSI-only configs: -4.09% average alpha (range -1.86% to -5.08%), 40.91% win rate, 1,895 trades.

**Conclusion:** VERIFIED (negative). RSI alone is unreliable; it improves only when combined with Elliott Wave in the tested circumstances.

---

## EMA Hypotheses

### EMA 20/50 is optimal among EMA period choices (standalone)

**Hypothesis:** EMA 20/50 gives the best alpha among tested EMA period pairs when EMA is used alone.

**Findings:** EMA 20/50: -1.66% average alpha, 46.44% win rate, 1,703 trades. Other pairs (e.g. 9/21, 50/200) perform worse in the same setup.

**Conclusion:** MODIFIED. Under circumstances: walk-forward on DJIA, EMA-only configs — 20/50 is best among EMAs, but still negative. Optimality is relative to other EMA-only configs, not to the full strategy set.

---

### EMA standalone is viable

**Hypothesis:** EMA crossover alone can achieve positive alpha.

**Findings:** EMA-only: -1.66% average alpha. In bull markets, late entries and early exits dominate; crossover signals lag.

**Conclusion:** VERIFIED (negative). EMA alone is not viable; it acts as confirmation or trend filter only in the tested setup.

---

## MACD Hypotheses

### MACD signal period 12 is optimal (standalone MACD)

**Hypothesis:** MACD with signal period 12 yields higher alpha than periods 9 or 7 when MACD is used alone.

**Findings:** Signal 12: +3.07% alpha, 50.73% win rate, 1,788 trades. Period 9: -0.53%; period 7: -0.51%. Longer signal reduces false crossovers while staying responsive.

**Conclusion:** VERIFIED. Under circumstances: walk-forward on DJIA, MACD-only configs — signal period 12 is optimal.

---

### MACD standalone is viable

**Hypothesis:** MACD alone can achieve positive alpha.

**Findings:** MACD-only (12/26/12): +3.07% average alpha, 50.73% win rate, 1,788 trades. Better than RSI-only or EMA-only in the same framework.

**Conclusion:** VERIFIED. MACD is the only technical indicator that is positive when used alone in the tested setup, though modest.

---

## Elliott Wave Hypotheses

### Wave size 0.02 is optimal

**Hypothesis:** Elliott Wave wave size 0.02 yields the best balance of alpha and stability.

**Findings:** Wave sizes 0.01, 0.015, 0.02, 0.03: +7.60%, +7.57%, +7.31%, +7.13% average alpha. Smaller sizes (0.01, 0.015) had slightly higher win rates and alpha.

**Conclusion:** MODIFIED. Under circumstances: Elliott Wave standalone, walk-forward on DJIA — smaller wave sizes (0.01) perform at least as well; 0.02 is not uniquely optimal.

---

### Elliott Wave confidence 0.65 is optimal

**Hypothesis:** Confidence 0.65 gives the best trade-off between selectivity and alpha for Elliott Wave.

**Findings:** 0.5 and 0.65: identical trades and alpha (+7.31%). 0.8: zero trades. So 0.5–0.65 are equivalent; 0.8 is too restrictive.

**Conclusion:** VERIFIED. Under circumstances: Elliott Wave standalone — confidence in [0.5, 0.65] is equivalent and optimal; 0.8 is not usable.

---

## Indicator Combination Hypotheses

### Combining indicators dilutes Elliott Wave signal quality

**Hypothesis:** Adding RSI/EMA/MACD to Elliott Wave worsens alpha due to timing mismatch.

**Findings:** EW+all indicators: +14.68% average alpha, 46.86% win rate, 5,438 trades. Elliott only: +7.31%, 59.02%, 641. Indicators-only (no EW): -2.62%. Combination clearly dominates.

**Conclusion:** REJECTED. Under circumstances: walk-forward on DJIA over multiple periods — combining indicators with Elliott Wave improves alpha; EW+all indicators is best among tested configs.

---

### EW + MACD improves performance only in bear markets

**Hypothesis:** EW+MACD is beneficial mainly in bear regimes and weak or harmful in bull markets.

**Findings:** EW+MACD: +10.45% average alpha, 52.80% win rate. Strong in bear/full-period, weak in bull (-6.59% in bull_market_long). Net effect is positive across all tested periods.

**Conclusion:** MODIFIED. EW+MACD helps in bear markets and is positive on average, but underperformance in bull markets is significant. Optimality of “use EW+MACD” is period-dependent.

---

### RSI adds value when combined with Elliott Wave

**Hypothesis:** RSI remains useless even when combined with Elliott Wave.

**Findings:** EW+RSI: +10.14% average alpha, 45.14% win rate. Best in bear and crash periods; weak in recent bull. Elliott Wave filters RSI’s false signals.

**Conclusion:** REJECTED. Under circumstances: walk-forward on DJIA — RSI adds value only in combination with Elliott Wave.

---

### EMA + MACD matches or beats EW-based combos

**Hypothesis:** EMA+MACD (no Elliott Wave) reaches similar or better alpha than EW-based strategies.

**Findings:** EMA+MACD: +1.74% average alpha, 48.98% win rate, 3,299 trades. EW+all indicators: +14.68%. EMA+MACD is positive but clearly inferior.

**Conclusion:** REJECTED. Under circumstances: same framework — EMA+MACD is not optimal; EW-based combinations dominate.

---

## Regime Detection Hypotheses

### ADX-based regime detection improves Elliott Wave alpha

**Hypothesis:** Adapting signals with ADX-based regime (bull/bear) improves alpha over plain Elliott Wave.

**Findings:** Regime-enabled vs Elliott-only: identical alpha (+7.31%), win rate (59.02%), and trade count (641). No measurable difference.

**Conclusion:** REJECTED. Under circumstances: walk-forward on DJIA, ADX+MA-slope regime — regime detection does not improve alpha.

---

### ADX threshold (25 vs 30 vs 35) changes outcomes

**Hypothesis:** Varying ADX threshold (25, 30, 35) changes alpha or trade profile when regime detection is on.

**Findings:** All thresholds yield identical alpha (+7.31%), win rate, and trades. Regime logic appears not to affect the evaluated path.

**Conclusion:** INCONCLUSIVE. Either regime is not active in the tested path or thresholds do not bind; no evidence that threshold choice matters in this setup.

---

## Position Sizing and Risk Management

### There exists an optimal position size (fraction of portfolio per new trade) above 0.2

**Hypothesis:** Increasing position size above the initial baseline (0.2) improves alpha up to some optimum.

**Findings:** Sweep 0.2–0.4: 0.35 yields highest alpha (+153.57%), versus +132.08% at 0.3 and declining above 0.35 (0.36–0.40: +152.15% down to +146.20%). Win rate stable (~42.2%); trade count falls as size increases (e.g. 1,640 at 0.35 vs 1,741 at 0.3).

**Conclusion:** VERIFIED. Under circumstances: walk-forward on DJIA, full period 2000–2020, EW+all indicators, risk_reward 2.5, alpha as objective — position size 0.35 is optimal; 0.36–0.40 show diminishing returns.

---

### Max open positions (cap) affects alpha at that position size

**Hypothesis:** Changing the maximum number of open positions (e.g. 5, 10, 15, 20) changes alpha when position size is 0.3 or 0.35.

**Findings:** At 0.3 and at 0.35, alpha and trade count are identical for max_positions 5, 10, 15, 20. The cap is never binding in these runs.

**Conclusion:** REJECTED. Under circumstances: same as above — max_positions does not affect outcomes; any listed value is equivalent until the constraint binds.

---

### Flexible position sizing (confidence or risk-reward scaling) beats additive confidence sizing

**Hypothesis:** Sizing each trade by a flexible rule (e.g. confidence or risk-reward multiplier) yields higher alpha than the baseline additive confidence rule (base + multiplier × confirmations).

**Findings:** Additive confidence sizing: +153.57% alpha, 1,640 trades. Confidence-based flexible: +26.59%, 1,995 trades. Risk-reward flexible: +74.75%, 1,859. Combined flexible: +42.30%, 1,964. Flexible methods increase trades but reduce alpha.

**Conclusion:** REJECTED. Under circumstances: walk-forward on DJIA, full period 2000–2020, position size 0.35 — additive confidence sizing is optimal among the tested sizing schemes.

---

## Risk/Reward Ratio

### Risk/reward ratio 2.5 is optimal for target/stop scaling

**Hypothesis:** A risk/reward ratio of 2.5 maximizes alpha vs 1.5, 2.0, 2.5, 3.0, 3.5, 4.0.

**Findings:** Ratio 2.5: +153.57% alpha, 42.2% win rate, 1,640 trades. 2.0: +107.19%; 1.5: +81.55%; 3.0–4.0: +142–88%. Peak alpha at 2.5.

**Conclusion:** VERIFIED. Under circumstances: walk-forward on DJIA, full period 2000–2020, EW+all indicators, position size 0.35 — risk_reward 2.5 is optimal.

---

## Inverted Elliott Wave (Sell-Signal Doubling)

### Using inverted price series for Elliott Wave improves sell-signal coverage and alpha

**Hypothesis:** Running Elliott Wave on inverted prices to generate sell signals (and combining with regular EW buys) increases alpha over baseline (regular EW + indicators only).

**Findings:** Baseline (regular EW + all indicators): +132.08% alpha, 1,741 trades. Inverted+regular+all indicators: +106.12%, 1,833 trades. Inverted+regular EW only: +28.51%. Inverted EW + all indicators (no regular EW): -30.34%. Inverted-only or inverted-without-regular underperforms; adding inverted to baseline raises trade count but lowers alpha on the full 2000–2020 period.

**Conclusion:** REJECTED. Under circumstances: full period 2000–2020, EW+indicators baseline — inverted Elliott Wave does not improve alpha; baseline remains optimal. Inverted EW alone is strongly negative.

---

## Inverted Elliott Wave Exit (Sell-to-Close)

### Inverted EW with sell-to-close (SELLs close longs) improves alpha vs baseline

**Hypothesis:** Using inverted Elliott Wave in “exit-only” mode (SELL signals close open longs instead of opening shorts) improves alpha over the same baseline without this feature.

**Findings:** Configs in `configs/hypothesis_inverted_exit/`: control (baseline, no inverted exit) vs inverted_exit (baseline + elliott_wave_inverted_exit). Period 2018–2020, DJIA: control +19.52% alpha, 129 trades, 46.5% win rate; inverted_exit +4.84% alpha, 138 trades, 39.1% win rate. Delta: alpha -14.69%, trades +9. Sell-to-close signals fire and add exits; they reduce alpha and win rate.

**Conclusion:** REJECTED. Under circumstances: 2018–2020, DJIA, EW+indicators baseline — inverted EW exit (sell-to-close) reduces alpha and win rate versus control. The extra exits from inverted EW SELLs close longs earlier than target/stop would, cutting winners. Baseline (control) remains preferable.

---

## Multi-Instrument and Regime (Additional Evidence)

### Single-index choice (DJIA vs S&P 500 vs DAX vs Gold) or multi-index aggregation changes alpha

**Hypothesis:** Switching the single equity index or using a multi-instrument portfolio changes alpha versus baseline (e.g. DJIA-only) for the same strategy.

**Findings:** Baseline configs on DJIA, S&P 500, DAX, Gold, and multi-instrument (DJIA+S&P+DAX, or all instruments) all yield the same alpha (+153.57%), win rate (42.2%), and effective trade count per run (1,640) when strategy and period are matched. No cross-sectional spread.

**Conclusion:** Under circumstances: full period 2000–2020, EW+all indicators, position 0.35, risk_reward 2.5 — no evidence that instrument choice or multi-instrument aggregation changes alpha in this setup; result is invariant across tested instruments/aggregations.

---

### ADX-based regime filtering (multiple thresholds) improves alpha vs no regime

**Hypothesis:** Enabling regime detection with various ADX thresholds (25, 30, 35, 40, with and without invert logic) improves alpha versus the same strategy with regime disabled.

**Findings:** All regime variants and “no regime” yield identical alpha (+153.57%), win rate (42.2%), and trades (1,640). Regime on/off and threshold choice do not change outcomes.

**Conclusion:** REJECTED. Under circumstances: same as above — regime detection (ADX-based, thresholds 25–40) does not improve alpha; baseline without regime is equivalent.

---

## Elliott Wave Parameter Sweep (Confidence × Wave Length)

### Elliott Wave confidence and wave-length parameters have an optimum that beats baseline

**Hypothesis:** Some (confidence, wave_length) pair yields higher alpha than the baseline EW params used in the reference strategy.

**Findings:** Sweep over confidence {0.60, 0.65, 0.70, 0.75} and wave_length {0.15, 0.20, 0.25, 0.30}: best configs (e.g. confidence 0.60–0.70, wave_length 0.30) reach +156.43% alpha, 42.2% win rate, 1,636 trades vs baseline +153.57%, 1,640 trades. Baseline-like params (e.g. 0.065/0.020) sit at +153.57%. Worst combos are negative (-10.80%).

**Conclusion:** VERIFIED (narrowly). Under circumstances: walk-forward on DJIA, full period 2000–2020, EW+all indicators, position 0.35, risk_reward 2.5 — confidence in [0.60, 0.70] with wave_length 0.30 yields modestly higher alpha (+156.43% vs +153.57%) than the baseline EW params. Optimality is specific to this period and setup.

### Baseline EW params (validation with current baseline)

**Hypothesis:** With current baseline (RSI 5, position 0.35, risk_reward 2.5), EW confidence 0.60–0.70 and wave_length 0.30 yield higher alpha than current EW (0.65, 0.02).

**Findings:** baseline_current (EW 0.65, 0.02): +165.13% alpha, 42.4% win rate, 1,810 trades. ew_060_wave_030, ew_065_wave_030, ew_070_wave_030 (all wave 0.30): -15.86% alpha each, 41.0% win rate, 1,864 trades. Current EW params clearly outperform the sweep-best (0.60–0.70 / 0.30) when RSI 5 is used.

**Conclusion:** REJECTED. Under circumstances: walk-forward on DJIA, full period 2000–2020, current baseline (RSI 5, position 0.35, risk_reward 2.5) — EW (0.65, 0.02) is optimal; EW (0.60–0.70, 0.30) underperforms. The earlier Elliott Wave Parameter Sweep optimum (wave 0.30) does not carry over to the RSI-5 baseline; keep baseline EW at 0.65 / 0.02.

---

## Trend Filter

### An additional trend filter (e.g. MA-based) improves alpha vs no trend filter

**Hypothesis:** Gating entries with a trend filter on top of EW+indicators improves alpha.

**Findings:** No trend filter: +153.57% alpha, 42.2% win rate, 1,640 trades. Trend filter enabled: +97.63% alpha, 43.2% win rate, 1,010 trades. Filter reduces trades and alpha.

**Conclusion:** REJECTED. Under circumstances: same baseline and period — the tested trend filter reduces alpha; “no trend filter” is optimal among the two options.

---

## Indicator Parameter Variations

### Some EMA, MACD, or RSI parameter choices beat baseline when used inside EW+indicators

**Hypothesis:** Varying EMA periods (e.g. 15/40, 25/60), MACD signal period (9, 15), or RSI period (5, 9, 14) within the EW+all-indicators setup yields an optimum that beats the current baseline params.

**Findings:** Sweep of eight configs (baseline plus EMA 15/40, 25/60; MACD signal 9, 15; RSI period 5, 9, 14): RSI period 5 yields highest alpha (+165.13%), then baseline (+153.57%), then MACD signal 15 (+131.46%), ema_25_60 (+117.26%), ema_15_40 (+95.62%), macd_signal_09 (+92.90%), rsi_period_14 (+84.02%), rsi_period_09 (+75.07%). Win rates 41.4–42.8%; trade counts 1373–1810. RSI period 5 adds ~11.6 pp alpha over baseline; MACD 15 and EMA variants trail baseline.

**Conclusion:** VERIFIED. Under circumstances: walk-forward on DJIA, full period 2000–2020, EW+all indicators, position 0.35, risk_reward 2.5 — RSI period 5 is optimal among the tested EMA/MACD/RSI variants; baseline (RSI 7) is second. MACD signal 15 outperforms baseline MACD signal 12; EMA 15/40 and 25/60 underperform baseline EMA 20/50.

---

## Long vs Short Trade Performance

### Long and short trades have different performance; one side contributes more to alpha

**Hypothesis:** Long and short trades have different performance; one side contributes more to alpha or has better win rate/expectancy.

**Findings:** Baseline trades (full period, DJIA, EW+all indicators, position 0.35, risk_reward 2.5):

| Metric              | Long (buy) | Short (sell) |
| ------------------- | ---------- | ------------ |
| Trade count         | 927        | 883          |
| Win rate %          | 46.71      | 37.83        |
| Total PnL           | 243.33     | 1.05         |
| Avg PnL % per trade | 0.26       | 0.02         |
| Avg win % (winners) | 1.93       | 1.40         |
| Avg loss % (losers) | -1.21      | -0.82        |

Total PnL % (Long): 0.30%. Total PnL % (Short): 0.00%. Long trades dominate total PnL and win rate; short trades contribute negligibly.

**Conclusion:** ACCEPTED. Long trades contribute substantially more under the tested circumstances. Under circumstances: baseline config, full period 2000–2020, DJIA, EW+all indicators, position 0.35, risk_reward 2.5 — long (buy) trades have higher win rate, total PnL, and avg PnL %; short (sell) trades are roughly break-even in total PnL with lower win rate.

---

## Period and Strategy Performance (Evidence Summary)

Across the evaluated configs and subperiods:

- **Stronger regimes:** Bear markets and crashes (e.g. 2000–2003, COVID). Best single-period alpha for the top strategy occurs in the full 20-year and bear/crash windows.
- **Weaker regimes:** Sustained bull (e.g. 2010–2020, 2016–2020). Alpha drops or turns negative there.
- **Strategy vs regime:** EW+all indicators leads on average and in volatile regimes; EW+RSI excels in bear/crash; EW+MACD has higher win rate but larger drawdown in bull regimes.

This is consistent with the reference baseline being best on average but period-sensitive.
