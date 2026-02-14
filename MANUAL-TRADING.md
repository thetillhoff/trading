# Manual Trading Guide

A step-by-step schedule for manual trading using the daily recommendation system with Interactive Brokers.

## Overview

This guide assumes you're:
- Trading US stocks (market hours: 9:30 AM - 4:00 PM ET)
- Using **Interactive Brokers** or **ING** (workflows for both provided)
- Following the baseline strategy recommendations
- Holding positions for 2-4 weeks (swing trading)

**Choose your broker section:**
- [Interactive Brokers Workflow](#interactive-brokers-workflow) - Automated bracket orders (recommended)
- [ING Workflow](#ing-workflow) - Manual order management

---

## Daily Schedule

### Evening (7:00-8:00 PM ET) - After Market Close

**Goal**: Update data and get tomorrow's recommendation

#### Step 1: Update Market Data (5 minutes)
```bash
cd ~/code/thetillhoff/trading
make download
```

**What this does:**
- Downloads latest closing prices for all instruments
- Updates ticker data from Yahoo Finance
- Ensures you have today's completed trading day

**Expected output:**
```
Updated 100/100 tickers
```

#### Step 2: Get Recommendation (2 minutes)
```bash
make recommend
```

**What to look for:**
```
Best opportunity:
  Instrument: CCL
  Action: BUY
  Price: $31.77
  Confidence: 1.00 (100%)
  Stop-loss: $31.04
  Target: $35.11
  Risk/Reward: 4.5
  Signal date: 2026-02-13
```

**Decision criteria:**
- ✅ **Confidence ≥ 70%**: Consider the trade
- ✅ **Risk/Reward ≥ 2.0**: Good risk management
- ✅ **Signal date is today**: Fresh signal
- ❌ **Signal date is >3 days old**: Stale, skip

#### Step 3: Research the Signal (10-15 minutes)

If signal looks promising:

1. **Check news**: Google `[TICKER] news` - any major events?
2. **Verify reasoning**: Does the Elliott Wave + indicator confluence make sense?
3. **Check chart**: Look at [Yahoo Finance chart](https://finance.yahoo.com) - does visual support the signal?
4. **Position size**: Calculate how many shares based on your capital:
   ```
   Risk per trade: $100 (example)
   Distance to stop: $31.77 - $31.04 = $0.73
   Shares = $100 / $0.73 = 137 shares
   
   Total capital needed: 137 × $31.77 = $4,352
   ```

5. **Record decision**: Keep a trading journal (see template below)

---

## Broker-Specific Workflows

The daily recommendation system provides: entry price, stop-loss, and target. How you implement these depends on your broker's capabilities.

### Interactive Brokers Workflow

**Advantages:**
- ✅ Bracket orders (entry + exits in one order)
- ✅ OCO (One-Cancels-Other) - automatic management
- ✅ Set once and forget
- ✅ Lowest risk of human error

#### Morning Order Placement (9:15-9:25 AM ET)

**In TWS (Trader Workstation):**

1. **Right-click ticker** → "Attach Bracket Order"

2. **Entry Order (Parent)**:
   ```
   Action: BUY
   Quantity: [calculated shares]
   Order Type: LMT
   Limit Price: $31.77
   Time in Force: DAY
   ```

3. **Bracket Settings Automatically Create**:
   ```
   Take Profit: LMT $35.11 (GTC)
   Stop Loss: STP $31.04 (GTC)
   ```

4. **Verify OCO Enabled** (checkbox should be checked)

5. **Submit** - All three orders transmit together

**What happens next:**
- Entry order waits for fill
- When entry fills → both exit orders activate automatically
- When target OR stop hits → other order auto-cancels
- **Zero additional work required**

**In IBKR Mobile App:**
1. Search ticker → Trade
2. Select "Bracket" from order type dropdown
3. Enter entry price, stop, and target
4. Submit

---

### ING Workflow

**Limitations:**
- ❌ No bracket orders
- ❌ No OCO linking
- ⚠️ Manual 3-step process required
- ⚠️ Must remember to place exits after entry fills

#### Morning Order Placement (9:15-9:25 AM ET)

**Step 1 - Place Entry Order:**

In ING app/website:
```
Order Type: Limietorder (Limit)
Action: Kopen (Buy)
Ticker: CCL
Quantity: [calculated shares]
Limit: $31.77
Validity: Dag (Day)
```

**Submit and WAIT**

#### After Entry Fills (Could be 9:31 AM or later)

**⚠️ CRITICAL: Set alarm/reminder to check for fill at 9:35 AM**

Once you see entry filled, **immediately** place both exit orders:

**Step 2 - Place Target Order:**
```
Order Type: Limietorder (Limit)
Action: Verkopen (Sell)
Ticker: CCL
Quantity: [same as entry]
Limit: $35.11
Validity: GTC (if available) or longest available
```

**Step 3 - Place Stop-Loss Order:**
```
Order Type: Stop-loss order
Action: Verkopen (Sell)
Ticker: CCL  
Quantity: [same as entry]
Stop Price: $31.04
Validity: GTC (if available) or longest available
```

**Step 4 - Verify Both Orders Active:**
- Check "Mijn orders" (My Orders)
- Confirm both SELL orders are listed
- Both should show same quantity as your position

---

#### ING Daily Checklist (Critical!)

Since ING doesn't link orders, you must manually manage:

**Daily (While Position Open):**
- [ ] 10:00 AM: Check position status
- [ ] Verify both exit orders still active
- [ ] If ING only allows DAY orders: re-enter exits daily

**When One Exit Triggers:**
- [ ] **Immediately cancel the other exit order**
- [ ] Example: If target fills at $35.11 → cancel stop-loss order at $31.04
- [ ] If stop fills at $31.04 → cancel target order at $35.11

**Failure to Cancel = Double Order Risk:**
```
Bad scenario:
- Target fills at $35.11 (you SELL 137 shares)
- You forget to cancel stop-loss
- Price drops to $31.04
- Stop triggers → you SELL ANOTHER 137 shares (SHORT position!)
- Now you're short 137 shares unintentionally
```

---

#### ING-Specific Safety Measures

**1. Set Phone Alerts:**
- Alert at $35.11 (target price)
- Alert at $31.04 (stop price)
- Alert at $31.77 (break-even)

**2. Calendar Reminders:**
```
Daily 10:00 AM: "Check CCL position and orders"
Daily 3:30 PM: "Verify CCL orders still active"
```

**3. Order Tracking Sheet:**
```
Position: CCL | Entry: $31.77 | Shares: 137
┌─────────────────────────────────────────┐
│ Exit Order 1: SELL 137 @ $35.11 ✓      │
│ Exit Order 2: SELL 137 @ $31.04 ✓      │
│ Status: Both Active                     │
│ Last Checked: 2026-02-14 10:05 AM       │
└─────────────────────────────────────────┘
```

**4. End-of-Day Routine:**
```bash
# Check if you need to update orders
# ING order validity may be limited
```

---

### ING vs Interactive Brokers: Time Comparison

**Interactive Brokers:**
- Order placement: 5 minutes (one-time setup)
- Daily maintenance: 0 minutes (fully automated)
- Exit management: 0 minutes (OCO handles it)
- **Total time per trade: 5 minutes**

**ING:**
- Order placement: 5 minutes (entry only)
- Wait and place exits: 10 minutes (after fill)
- Daily check: 5 minutes × 10-20 days = 50-100 minutes
- Exit management: 5 minutes (manual cancel)
- **Total time per trade: 70-120 minutes**

---

## Choosing Your Broker for This System

### Use Interactive Brokers If:
- ✅ You want true "set and forget" trading
- ✅ You're managing multiple positions simultaneously
- ✅ You value automation and reduced stress
- ✅ You can handle slightly more complex platform

### Use ING If:
- ✅ You already have account and don't want to switch
- ✅ You're only taking 1-2 positions at a time
- ✅ You can commit to daily order checks
- ✅ You're disciplined about manual processes

### Don't Use Trade Republic For:
- ❌ Swing trades (2-4 weeks) - too much daily order renewal
- ⚠️ Only suitable for day trading or very short holds

---

## ING Pro Tips

**1. Mobile App Notifications:**
Enable push notifications for:
- Order fills
- Position changes
- Price alerts

**2. Order Templates:**
If ING supports it, save order templates:
- "Exit Long - Target" (pre-filled SELL limit)
- "Exit Long - Stop" (pre-filled SELL stop)

**3. Position Size:**
Start with smaller positions (1-2 positions max) until you're comfortable with the manual workflow

**4. Weekend Preparation:**
```
Saturday: Run make recommend
Sunday: Research signal, calculate position size
Monday 9:15 AM: Ready to place orders
```

**5. Trading Journal Critical:**
With manual management, your journal must include:
```
Order Status:
- Entry order placed: ✓
- Entry filled: ✓  
- Target order placed: ✓
- Stop order placed: ✓
- Both exits verified active: ✓
- Exit triggered: ___________
- Opposite exit cancelled: ___________
```

---

## Interactive Brokers Account Setup (Europe)

If you decide IBKR is worth it:

**Account Opening:**
1. Go to [IBKR Europe](https://www.interactivebrokers.eu)
2. Choose "Individual Account"
3. Complete application (~30 minutes)
4. Verification: Passport + proof of address
5. Approval: Usually 1-3 business days
6. Fund account: Bank transfer from ING (free)

**Minimum Requirements:**
- No minimum deposit (but need capital for trades)
- Knowledge test: Basic trading questions (easy)
- Must confirm you understand risks

**Monthly Costs:**
- Commission: ~$0.35-1.00 per trade (cheaper for high volume)
- No monthly fees if trading regularly
- Data feeds: Optional, $4.50/month for real-time quotes (not needed for this system)

**Time to Trading:**
- Application: 30 minutes
- Approval: 1-3 days
- Funding: 1-2 days
- **Total: ~1 week to start trading**

---

### During Market Hours (9:30 AM - 4:00 PM ET)

#### Opening (9:30-10:00 AM) - High Volatility
- **Monitor entry fill**: Check if your limit order filled
- **Adjust if needed**: If price runs away, decide whether to chase or skip
- **Confirm exits placed**: Once entry fills, ensure both exit orders are active

#### Mid-Day (10:00 AM - 3:00 PM) - Normal Activity
- **Minimal checking**: Avoid emotional trading
- **Set price alerts**: 
  - Alert at target price ($35.11)
  - Alert at stop price ($31.04)
  - Alert at break-even ($31.77) if you want to trail stop

#### Close (3:30-4:00 PM) - End of Day
- **Review positions**: Check if any exits triggered
- **No panic**: Don't close profitable positions just because market dips
- **Update journal**: Note any significant price action

---

### Weekly Tasks

#### Weekend (Saturday or Sunday Morning) - 30 minutes

1. **Review open positions**:
   ```bash
   # Check if signals still valid
   make recommend  # See if new signals appeared
   ```

2. **Update trading journal**:
   - Current P&L for each position
   - Days held
   - Any adjustments needed?

3. **Check for new signals**:
   - If no open positions, prepare for Monday
   - Research any new recommendations

4. **Adjust stops if needed** (optional):
   - If position moved 50% to target, consider moving stop to break-even
   - Example: CCL at $33.50 → move stop from $31.04 to $31.77

---

## Trading Journal Template

Keep a simple spreadsheet or notebook:

```
Date: 2026-02-14
Signal Date: 2026-02-13
Ticker: CCL
Action: BUY
Entry: $31.77
Stop: $31.04
Target: $35.11
Risk/Reward: 4.5
Confidence: 100%
Shares: 137
Capital: $4,352
Risk: $100 (137 × $0.73)

Reasoning: End of Wave 4 correction | RSI=29 (favorable) | EMA bullish | MACD bullish

News Check: Cruise industry recovering, no major concerns
Chart Check: Confirmed Wave 4 low, uptrend intact

Exit Date: ___________
Exit Price: ___________
P&L: ___________
Notes: ___________
```

---

## Position Management Rules

### When to Exit Early (Before Target/Stop)

**Exit immediately if:**
- Company announces bankruptcy/investigation
- Sector-wide crisis (e.g., COVID cruise shutdown)
- Technical break: Daily close below stop-loss by >2%

**Consider exiting if:**
- New SELL signal appears: `make recommend` shows SELL for same ticker
- Position held >6 weeks with no progress
- Market crashes (S&P 500 down >5% in one day)

### When to Hold

**Keep holding if:**
- Price consolidating near entry (normal for first week)
- Small moves against you (still above stop)
- Target not reached but trend still intact
- No new contradictory signals

### Position Sizing Guidelines

**Conservative (Recommended for beginners):**
- Risk 1% of capital per trade
- Example: $50k account → risk $500 per trade
- If CCL stop is $0.73 away → 684 shares × $31.77 = $21,730 position
- Only enter if you have at least $25k available

**Aggressive (Experienced traders):**
- Risk 2% of capital per trade
- $50k account → risk $1,000 per trade
- Allows larger positions but higher stress

**Never risk more than 2% per trade!**

---

## Monthly Tasks

### Month-End Review (1 hour)

1. **Calculate performance**:
   - Win rate: X wins / Y total trades
   - Average R:R achieved vs planned
   - Total P&L for month

2. **Strategy adjustment**:
   - Are signals profitable? Track hit rate
   - Do you need to be more/less selective?
   - Review journal for patterns

3. **System maintenance**:
   ```bash
   # Clean up old data if needed
   # No specific cleanup needed, system manages cache
   ```

---

## Troubleshooting

### "No signals detected today"

**Causes:**
- Market conditions don't meet criteria
- All instruments overbought/oversold
- No clear Elliott Wave patterns

**Action:**
- Wait for next day
- Don't force trades
- Cash is a position

### "Data is X days old"

**Cause:**
- Ticker not actively traded
- Market holiday
- Yahoo Finance delay

**Action:**
- Skip that ticker
- Run `make download` again
- Check if market was open

### "Signal from 3+ days ago"

**Cause:**
- You ran recommend without updating data first

**Action:**
```bash
make download  # Update first
make recommend # Then get fresh recommendation
```

### Signal price different from current market price

**Normal!** Signals are generated at previous day's close. Market may have:
- Gapped overnight
- Moved during the day

**Action:**
- If price > entry + 2%: Skip or use current price with adjusted stop
- If price < entry: Lucky! Better entry price
- Use limit orders to control entry

---

## Interactive Brokers Tips

### TWS Shortcuts
- **F12**: Quick trade ticket
- **Ctrl+N**: New order
- **Ctrl+F8**: Portfolio view
- **Alt+M**: Market depth

### Mobile App
- Use "Bracket Order" under order types
- Set up price alerts: Ticker → More → Alert
- Check positions: Portfolio → Positions

### Risk Management Features
- **Max % of Portfolio**: Settings → Trading → Risk Management
- **Daily Loss Limit**: Set to stop trading if down X%
- **Pattern Day Trader**: Need $25k to day trade (swing trading doesn't apply)

### Paper Trading
- Test this system first with paper account
- IB provides free paper trading: [Account Management](https://www.interactivebrokers.com)
- Run for 1-2 months before using real money

---

## Emergency Procedures

### Market Crashes (S&P 500 down >3%)
1. **Don't panic sell**
2. Check stop-losses still active
3. Consider tightening stops if position profitable
4. Let mechanical rules handle exits

### System/Internet Down
1. **Have broker phone number**: Call to close positions
2. **Know your positions**: Keep journal backed up
3. **Can use broker website**: Don't rely only on TWS

### Account Locked/Restricted
1. **Contact IB immediately**: 877-442-2757 (US)
2. **Have backup broker**: Consider keeping small balance elsewhere
3. **Document everything**: Screenshots of positions

---

## Success Checklist

Daily (Weekdays):
- [ ] 7:00 PM: Run `make download`
- [ ] 7:10 PM: Run `make recommend`
- [ ] 7:15 PM: Research signal if promising
- [ ] 7:30 PM: Decide on trade (yes/no)
- [ ] 9:15 AM: Review overnight news
- [ ] 9:25 AM: Place bracket order
- [ ] 9:35 AM: Confirm entry filled
- [ ] Update trading journal

Weekly:
- [ ] Review open positions
- [ ] Check for exit opportunities
- [ ] Adjust trailing stops if appropriate

Monthly:
- [ ] Calculate win rate and P&L
- [ ] Review journal for patterns
- [ ] Adjust strategy if needed

---

## Further Resources

**Trading Education:**
- Interactive Brokers University: [ibkr.com/university](https://www.interactivebrokers.com/university)
- Elliott Wave Theory: Robert Prechter's books
- Risk Management: "Trade Your Way to Financial Freedom" by Van Tharp

**System Documentation:**
- `README.md`: Overview of the trading system
- `DEVELOPMENT.md`: Technical architecture
- `HYPOTHESIS_TEST_RESULTS.md`: Strategy performance history

**Support:**
- Interactive Brokers Support: 877-442-2757
- Check `make help` for all available commands
- Review code in `cli/recommend.py` for how signals are generated

---

## Legal Disclaimer

This guide is for educational purposes only. Trading involves risk of loss. Past performance does not guarantee future results. Always:
- Start with paper trading
- Risk only capital you can afford to lose
- Understand tax implications (consult a tax professional)
- Comply with pattern day trader rules ($25k minimum for day trading)
- Read broker disclosures carefully

The recommendation system is based on historical backtesting but does not guarantee profits. You are responsible for your own trading decisions.
