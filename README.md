# H&S / Inverse H&S Pattern Scanner

Standalone module — kept separate from MaverickPICKS main_v3 as agreed.

## Files
- `hs_detector.py` — core detection engine (pivots, geometry matching, volume gate, target/SL)
- `fetch_data.py` — yfinance data fetch layer (same source as MaverickPICKS)
- `batch_scan.py` — batch runner: scans a symbol list, writes CSV output

## How to run

**Full scan (everything, all statuses):**
```bash
python batch_scan.py --symbols nifty500.csv --lookback 365 --out hs_scan_results.csv
```

**Watchlist mode (recommended for daily use) — only actionable, forward-looking setups:**
```bash
python batch_scan.py --symbols nifty500.csv --lookback 365 --out hs_watchlist.csv --watchlist
```
This excludes stale Confirmed patterns (breakout >10 trading days ago, or target already
hit), Failed patterns, and False Starts. Output is sorted by Quality Score, highest first.

- `--symbols` — CSV file with a `Symbol` column. **Reuse your existing MaverickPICKS
  NIFTY500 list file** — just point this at it. Symbols should be bare (e.g. `RELIANCE`,
  not `RELIANCE.NS` — the `.NS` suffix is added automatically.
- `--lookback` — days of price history to fetch (default 365)
- `--out` — output CSV path (default `hs_scan_results.csv`)
- `--only-confirmed` — only write Confirmed patterns (ignored if `--watchlist` is set)
- `--watchlist` — only write actionable setups (see above), sorted by Quality Score

## Output columns
Symbol, Pattern Type, Status, **Watchlist Category**, Left/Head/Right Shoulder dates &
prices, **RS Confirmation**, Neckline Price, Breakout Date/Price, Volume Confirmed,
**Breakout Volume Ratio**, Target, Stop Loss, Risk:Reward, **Target Hit**,
**Quality Score**, **Trigger Condition**, Notes

### New fields explained
- **Quality Score (0-100)** — a SHAPE-FIDELITY score: how closely the pattern matches
  Murphy's textbook criteria (depth symmetry, time symmetry, head prominence, neckline
  flatness, decline-phase volume profile). **This is NOT a win-probability or backtested
  confidence number** — that would require running the out-of-sample historical backtest
  we haven't done yet. Treat it as "how clean is this shape," not "how likely is this
  trade to work."
- **Breakout Volume Ratio** — breakout-day volume ÷ trailing 20-day average. A 1.5
  barely clears our gate; a 10-30x ratio (like the real WOCKPHARMA case that validated
  this tool) is a much stronger, more unambiguous signal. Lets you distinguish marginal
  from emphatic breakouts instead of treating all "Volume Confirmed: Yes" rows as equal.
- **RS Confirmation** — `Confirmed` (right shoulder passed the full 5-day fractal
  validation) or `Provisional` (a potential right shoulder in the most recent bars that
  hasn't had enough time to fully confirm yet — could still be undone by tomorrow's
  candle). Provisional patterns let you see a setup forming a few days earlier, at the
  cost of being less certain.
- **Watchlist Category** — `Watching - Confirmed RS` (pattern complete, waiting on
  breakout), `Watching - Provisional RS` (early heads-up, RS not fully confirmed yet),
  `Recent Breakout` (confirmed within the last 10 trading days, target not yet hit), or
  blank (excluded — stale, failed, or already played out).
- **Trigger Condition** — for pre-breakout patterns, the exact price level and share
  volume needed to confirm, e.g. *"Confirms if daily close goes above 158.05 with
  volume >= 216,324 shares (~1.5x 20-day avg)"*.
- **Target Hit** — for Confirmed patterns, whether price has already reached the
  projected target. Used internally to exclude stale wins from the watchlist.
- **RS Age (days) / Is Stale** — for Forming patterns, how many trading days have
  passed since the right shoulder formed, and whether that exceeds the adaptive
  staleness threshold (1.5x the pattern's own average leg duration). Stale patterns
  still appear in `--watchlist` output but are clearly labeled `Stale - ...` so you
  can deprioritize them rather than treating them as fresh setups.
- **`Invalidated` status** — price closed back past the right shoulder level before
  the neckline was ever broken, which breaks the pattern's structural validity per
  Murphy's criteria. These are excluded from the watchlist entirely (unlike Stale,
  which still shows up but flagged).

## Status so far
- Detection logic validated on synthetic data (correctly identifies shape, dates, levels)
- **Bug fixed (found via real NIFTY500 data, 3MINDIA):** candidate generation was
  combinatorial — testing every later trough as an alternate right shoulder for a
  given head, and every later trough as an alternate head for a given left shoulder.
  This produced multiple overlapping "patterns" that were really one underlying
  structure sliced different ways (same neckline, same breakout, different
  head/right-shoulder pivot picked). Fixed by switching to sequential consecutive-pivot
  matching (low-high-low-high-low / high-low-high-low-high in order), matching how a
  chartist actually reads a chart once, left to right.
- This fix also resolved a related issue: one of the over-generated candidates had a
  time-symmetry ratio of 2.3x — above our own TIME_RATIO_MAX of 2.2 — that should have
  been rejected but wasn't reachable as a clean gate inside the old combinatorial search.
- Re-ran null-hypothesis test after the fix: candidate count on 200 random-walk stocks
  dropped from 665 → 38 (~94% reduction), and Confirmed-on-noise rate dropped from
  3.0% → 0.0%.
- **Real-data spot checks (manual, against live charts):**
  - 3MINDIA — confirmed the over-generation bug (5 duplicate rows, one violating our
    own time-symmetry filter)
  - SWIGGY — geometry and volume gate both legitimate; pattern worked initially then
    lost momentum post-breakout (a real, if unflattering, outcome — not a detection bug)
  - WOCKPHARMA — initially suspected as another bug from chart-reading, but raw data
    confirmed the right shoulder price (1382 on 4/30) was real; breakout volume was a
    ~30x spike, an unambiguous signal; pattern went on to exceed its target. This case
    motivated adding the Breakout Volume Ratio field, since a 1.5x and 30x spike were
    being treated identically before.
- **New features added (status/quality/watchlist layer):**
  - Quality Score (0-100) — shape-fidelity composite, explicitly NOT a backtested
    win-probability (that requires the out-of-sample backtest, still pending)
  - Breakout Volume Ratio — numeric conviction measure instead of binary Yes/No
  - Provisional right-shoulder detection — surfaces early, not-yet-fractal-confirmed
    setups, clearly separated from fully Confirmed ones
  - Watchlist mode (`--watchlist`) — only actionable setups (pre-breakout +
    recent breakouts), excludes stale/Failed/False Start, sorted by Quality Score
  - New `False Start` status — separates "neckline crossed without volume
    confirmation" from genuine still-forming patterns (previously both were
    lumped into `Forming`, which understated how filtered the real watchlist is)
  - **New `Invalidated` status** — catches a real gap: a Forming pattern was
    previously left "waiting for breakout" forever even if price had already
    closed back past the right shoulder level (which breaks Murphy's structure
    entirely). Now checked explicitly: if invalidation happens before any neckline
    break, status is set to `Invalidated` and any later neckline cross is ignored
    (it wouldn't validate the original, already-broken structure). On the
    null-hypothesis test, this caught 10 of the 38 random-walk candidates that
    were previously sitting in `Forming` undetected.
  - **Adaptive staleness flagging** — a Forming pattern waiting longer than
    `STALENESS_MULTIPLIER` (1.5x) its own average leg duration for a breakout gets
    flagged `Stale - ... (overdue, Nd since RS)` rather than silently treated as
    fresh. Scales per-pattern (a 6-month formation gets more patience than a
    3-week one) rather than using one fixed day count for every stock.
- Three real bugs caught and fixed total during testing:
  1. Rolling 20-day volume average was including the breakout day in its own baseline
  2. Head-leg volume was being averaged across the full leg instead of just the decline phase
  3. Combinatorial over-generation of duplicate/overlapping pattern candidates
- Full pipeline (fetch → detect → CSV, including watchlist mode) tested end to end with
  mocked data — confirmed stale patterns get excluded and genuine pre-breakout setups
  appear correctly with trigger conditions

## NOT yet done — needs your real data
- `DEPTH_TOLERANCE` (0.18) and `TIME_RATIO_MAX` (2.2) are literature-based starting
  values, not yet tuned against real NIFTY500 history. Run this against your actual
  list and send me a sample of the output (especially Confirmed and Forming-but-close
  rows) so we can sanity-check whether these thresholds are too loose or too strict
  for real NSE price behavior.
- No out-of-sample backtest yet (does a Confirmed pattern actually tend to hit its
  target historically?) — that's the next phase once thresholds are settled.
- This is a "decision-support" tool, not a trade signal — treat Confirmed patterns
  as a watchlist flag to combine with your own judgment, same as we discussed.

## Tunable parameters (top of hs_detector.py)
PIVOT_WINDOW, DEPTH_TOLERANCE, TIME_RATIO_MAX, MIN_HEAD_PROMINENCE,
NECKLINE_SLOPE_MAX, VOLUME_BREAKOUT_MULT, VOLUME_HEAD_DECLINE_OK, DECLINE_LOOKBACK
