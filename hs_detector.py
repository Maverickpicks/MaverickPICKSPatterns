"""
Head & Shoulders / Inverse Head & Shoulders Pattern Detector
================================================================
Standalone module — deliberately kept separate from MaverickPICKS main_v3.

Pipeline:
    1. Pivot (swing high/low) detection using fractal logic
    2. Candidate H&S / Inverse H&S shape matching with symmetry + time tolerances
    3. Volume confirmation gate (Murphy's criteria)
    4. Status classification: Forming / Confirmed / Failed
    5. Target / Stop-Loss / R:R calculation for Confirmed patterns

This module contains NO data-fetching logic. It operates on a pandas
DataFrame with columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
sorted ascending by Date. Plug in your own data source (yfinance, NSE,
or your existing MaverickPICKS pipeline) via fetch_data.py.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ----------------------------------------------------------------------
# Tunable parameters (kept explicit and visible — not hidden magic numbers)
# ----------------------------------------------------------------------
PIVOT_WINDOW = 5            # candles on each side to qualify as a swing pivot
DEPTH_TOLERANCE = 0.18      # max allowed % difference between shoulder depths
TIME_RATIO_MAX = 2.2        # max allowed ratio between (head-to-LS) and (RS-to-head) durations
MIN_HEAD_PROMINENCE = 0.03  # head must be at least 3% deeper/higher than shoulders
NECKLINE_SLOPE_MAX = 0.15   # max allowed neckline slope as % of price over its span
VOLUME_BREAKOUT_MULT = 1.5  # breakout volume must be >= 1.5x the 20-day avg
VOLUME_HEAD_DECLINE_OK = 1.2  # head-leg decline volume can be up to 20% higher than left-shoulder decline volume and still pass

PROVISIONAL_PIVOT_WINDOW = 2   # relaxed window used ONLY to surface an early, not-yet-confirmed right shoulder candidate
RECENT_BREAKOUT_DAYS = 10      # a Confirmed breakout within this many trading days still counts as "actionable", not stale
HEAD_PROMINENCE_FULL_SCORE = 0.20  # prominence at/above this level scores 100 on the quality sub-score (diminishing returns above this)
STALENESS_MULTIPLIER = 1.5     # a Forming pattern waiting longer than (avg leg duration * this) for breakout is flagged overdue

# Quality score weights (must sum to 1.0) — purely a measure of how cleanly the shape
# matches Murphy's textbook criteria. NOT a statistical confidence/probability — that
# would require an out-of-sample backtest we haven't run yet.
QUALITY_WEIGHTS = {
    'depth_symmetry': 0.25,
    'time_symmetry': 0.20,
    'head_prominence': 0.15,
    'neckline_slope': 0.15,
    'volume_profile': 0.25,
}


@dataclass
class Pivot:
    index: int
    date: pd.Timestamp
    price: float
    kind: str  # 'high' or 'low'


@dataclass
class PatternResult:
    symbol: str
    pattern_type: str        # 'H&S' or 'Inverse H&S'
    status: str               # 'Forming' / 'Confirmed' / 'Failed' / 'False Start' / 'Invalidated'
    left_shoulder_date: pd.Timestamp
    left_shoulder_price: float
    head_date: pd.Timestamp
    head_price: float
    right_shoulder_date: pd.Timestamp
    right_shoulder_price: float
    neckline_price: float
    breakout_date: Optional[pd.Timestamp] = None
    breakout_price: Optional[float] = None
    volume_confirmed: bool = False
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None
    notes: str = ""
    quality_score: Optional[float] = None        # 0-100, shape-fidelity score — NOT a win-probability
    breakout_volume_ratio: Optional[float] = None  # breakout day volume / trailing 20-day avg (only once breakout occurs)
    rs_confirmation: str = "Confirmed"            # 'Confirmed' (5-day fractal validated) or 'Provisional' (early, unconfirmed)
    target_hit: bool = False
    trigger_condition: Optional[str] = None       # human-readable condition for pre-breakout patterns
    watchlist_category: Optional[str] = None      # 'Watching - Confirmed RS' / 'Watching - Provisional RS' / 'Recent Breakout' / 'Stale...' / None
    is_stale: bool = False
    rs_age_days: Optional[int] = None             # trading days elapsed since right shoulder, for Forming patterns


# ----------------------------------------------------------------------
# Step 1: Pivot detection (fractal swing highs/lows)
# ----------------------------------------------------------------------
def find_pivots(df: pd.DataFrame, window: int = PIVOT_WINDOW) -> List[Pivot]:
    pivots = []
    highs = df['High'].values
    lows = df['Low'].values
    dates = df['Date'].values
    n = len(df)

    for i in range(window, n - window):
        local_high_slice = highs[i - window:i + window + 1]
        local_low_slice = lows[i - window:i + window + 1]

        if highs[i] == local_high_slice.max() and (local_high_slice == highs[i]).sum() == 1:
            pivots.append(Pivot(index=i, date=pd.Timestamp(dates[i]), price=highs[i], kind='high'))
        if lows[i] == local_low_slice.min() and (local_low_slice == lows[i]).sum() == 1:
            pivots.append(Pivot(index=i, date=pd.Timestamp(dates[i]), price=lows[i], kind='low'))

    pivots.sort(key=lambda p: p.index)
    return pivots


# ----------------------------------------------------------------------
# Step 2: Candidate shape matching
# ----------------------------------------------------------------------
def _depth_symmetry_ok(left_val: float, right_val: float) -> bool:
    avg = (abs(left_val) + abs(right_val)) / 2
    if avg == 0:
        return False
    diff_ratio = abs(left_val - right_val) / avg
    return diff_ratio <= DEPTH_TOLERANCE


def _time_symmetry_ok(left_span: int, right_span: int) -> bool:
    if left_span <= 0 or right_span <= 0:
        return False
    ratio = max(left_span, right_span) / min(left_span, right_span)
    return ratio <= TIME_RATIO_MAX


def _neckline_slope_ok(p1: Pivot, p2: Pivot) -> bool:
    if p1.price == 0:
        return False
    slope_pct = abs(p2.price - p1.price) / p1.price
    return slope_pct <= NECKLINE_SLOPE_MAX


def compute_quality_score(depth_diff_ratio: float, time_ratio: float, head_prom_pct: float,
                           neckline_slope_pct: float, head_decline_vol: float,
                           ls_decline_vol: float) -> float:
    """
    0-100 composite score measuring how closely the shape matches Murphy's textbook
    criteria. This is a SHAPE-FIDELITY score, not a statistical win-probability —
    that would require backtesting historical outcomes, which we haven't done yet.
    """
    depth_score = max(0.0, 100 * (1 - depth_diff_ratio / DEPTH_TOLERANCE))

    time_excess = max(0.0, time_ratio - 1.0)
    time_score = max(0.0, 100 * (1 - time_excess / (TIME_RATIO_MAX - 1.0)))

    head_prom_score = max(0.0, min(100.0, 100 * (head_prom_pct - MIN_HEAD_PROMINENCE) /
                                    (HEAD_PROMINENCE_FULL_SCORE - MIN_HEAD_PROMINENCE)))

    neckline_score = max(0.0, 100 * (1 - neckline_slope_pct / NECKLINE_SLOPE_MAX))

    if ls_decline_vol and ls_decline_vol > 0:
        vol_ratio = head_decline_vol / ls_decline_vol
        if vol_ratio <= 1.0:
            volume_score = 100.0
        else:
            volume_score = max(0.0, 100 * (VOLUME_HEAD_DECLINE_OK - vol_ratio) / (VOLUME_HEAD_DECLINE_OK - 1.0))
    else:
        volume_score = 0.0

    total = (
        QUALITY_WEIGHTS['depth_symmetry'] * depth_score +
        QUALITY_WEIGHTS['time_symmetry'] * time_score +
        QUALITY_WEIGHTS['head_prominence'] * head_prom_score +
        QUALITY_WEIGHTS['neckline_slope'] * neckline_score +
        QUALITY_WEIGHTS['volume_profile'] * volume_score
    )
    return round(total, 1)


def find_inverse_hs_candidates(pivots: List[Pivot], symbol: str) -> List[PatternResult]:
    """
    Inverse H&S: trough(LS) -> peak -> trough(Head, lower) -> peak -> trough(RS, higher than head, ~= LS)

    IMPORTANT: only CONSECUTIVE pivots in sequence are considered (low-high-low-high-low),
    not every combination of later lows as alternate right-shoulder candidates. A real
    chartist reads the chart once, left to right — testing every later trough as an
    alternate right shoulder produces many overlapping pseudo-duplicates describing the
    same underlying structure (this was a real bug found via real NSE data).
    """
    candidates = []
    n = len(pivots)

    for i in range(n - 4):
        window = pivots[i:i + 5]
        kinds = [p.kind for p in window]
        if kinds != ['low', 'high', 'low', 'high', 'low']:
            continue

        ls, peak1, head, peak2, rs = window

        if head.price >= ls.price * (1 - MIN_HEAD_PROMINENCE):
            continue  # head not meaningfully lower than left shoulder
        if rs.price <= head.price * (1 + MIN_HEAD_PROMINENCE):
            continue  # right shoulder not meaningfully higher than head

        ls_depth = ls.price - head.price
        rs_depth = rs.price - head.price
        if not _depth_symmetry_ok(ls_depth, rs_depth):
            continue

        if not _time_symmetry_ok(head.index - ls.index, rs.index - head.index):
            continue

        if not _neckline_slope_ok(peak1, peak2):
            continue

        neckline_price = (peak1.price + peak2.price) / 2

        candidates.append(PatternResult(
            symbol=symbol,
            pattern_type='Inverse H&S',
            status='Forming',
            left_shoulder_date=ls.date, left_shoulder_price=ls.price,
            head_date=head.date, head_price=head.price,
            right_shoulder_date=rs.date, right_shoulder_price=rs.price,
            neckline_price=neckline_price,
        ))
    return candidates


def find_hs_candidates(pivots: List[Pivot], symbol: str) -> List[PatternResult]:
    """
    Regular H&S (topping): peak(LS) -> trough -> peak(Head, higher) -> trough -> peak(RS, lower than head, ~= LS)
    Only CONSECUTIVE pivots considered — see note in find_inverse_hs_candidates.
    """
    candidates = []
    n = len(pivots)

    for i in range(n - 4):
        window = pivots[i:i + 5]
        kinds = [p.kind for p in window]
        if kinds != ['high', 'low', 'high', 'low', 'high']:
            continue

        ls, trough1, head, trough2, rs = window

        if head.price <= ls.price * (1 + MIN_HEAD_PROMINENCE):
            continue
        if rs.price >= head.price * (1 - MIN_HEAD_PROMINENCE):
            continue

        ls_prom = head.price - ls.price
        rs_prom = head.price - rs.price
        if not _depth_symmetry_ok(ls_prom, rs_prom):
            continue

        if not _time_symmetry_ok(head.index - ls.index, rs.index - head.index):
            continue

        if not _neckline_slope_ok(trough1, trough2):
            continue

        neckline_price = (trough1.price + trough2.price) / 2

        candidates.append(PatternResult(
            symbol=symbol,
            pattern_type='H&S',
            status='Forming',
            left_shoulder_date=ls.date, left_shoulder_price=ls.price,
            head_date=head.date, head_price=head.price,
            right_shoulder_date=rs.date, right_shoulder_price=rs.price,
            neckline_price=neckline_price,
        ))
    return candidates


# ----------------------------------------------------------------------
# Step 3: Volume confirmation + Step 4: status classification
# ----------------------------------------------------------------------
def _avg_volume_in_range(df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    seg = df.iloc[max(0, start_idx):end_idx + 1]
    return seg['Volume'].mean() if len(seg) else np.nan


DECLINE_LOOKBACK = 10  # bars used to approximate the down-leg volume into a trough/peak when no prior pivot is available


def evaluate_pattern(df: pd.DataFrame, pattern: PatternResult, date_to_idx: dict) -> PatternResult:
    """Applies volume gate, checks for breakout/confirmation, computes target/SL/quality score."""
    ls_idx = date_to_idx[pattern.left_shoulder_date]
    head_idx = date_to_idx[pattern.head_date]
    rs_idx = date_to_idx[pattern.right_shoulder_date]
    is_inverse = pattern.pattern_type == 'Inverse H&S'

    # Decline-phase volume into the head: from the neckline peak/trough right before
    # the head, to the head itself — NOT the full leg (which would dilute with the
    # recovery rally on the other side).
    if is_inverse:
        peak1_idx = df['High'].iloc[ls_idx:head_idx + 1].idxmax()
        peak2_idx = df['High'].iloc[head_idx:rs_idx + 1].idxmax()
        peak1_price = df['High'].iloc[peak1_idx]
        peak2_price = df['High'].iloc[peak2_idx]
    else:
        peak1_idx = df['Low'].iloc[ls_idx:head_idx + 1].idxmin()
        peak2_idx = df['Low'].iloc[head_idx:rs_idx + 1].idxmin()
        peak1_price = df['Low'].iloc[peak1_idx]
        peak2_price = df['Low'].iloc[peak2_idx]

    head_decline_vol = _avg_volume_in_range(df, peak1_idx, head_idx)

    # Decline-phase volume into the left shoulder: approximated using a fixed
    # lookback window immediately preceding it, since there's no guaranteed prior
    # pivot in-sample to anchor on.
    ls_decline_start = max(0, ls_idx - DECLINE_LOOKBACK)
    ls_decline_vol = _avg_volume_in_range(df, ls_decline_start, ls_idx)

    # Murphy criterion: volume on the head's decline should not exceed the left
    # shoulder's decline volume (selling/buying pressure should be weakening into the head)
    head_volume_ok = (head_decline_vol <= ls_decline_vol * VOLUME_HEAD_DECLINE_OK) if ls_decline_vol else False

    # --- Quality score (shape-fidelity, independent of breakout outcome) ---
    if is_inverse:
        ls_depth = pattern.left_shoulder_price - pattern.head_price
        rs_depth = pattern.right_shoulder_price - pattern.head_price
        head_prom_pct = ((pattern.left_shoulder_price - pattern.head_price) / pattern.left_shoulder_price +
                          (pattern.right_shoulder_price - pattern.head_price) / pattern.right_shoulder_price) / 2
    else:
        ls_depth = pattern.head_price - pattern.left_shoulder_price
        rs_depth = pattern.head_price - pattern.right_shoulder_price
        head_prom_pct = ((pattern.head_price - pattern.left_shoulder_price) / pattern.left_shoulder_price +
                          (pattern.head_price - pattern.right_shoulder_price) / pattern.right_shoulder_price) / 2

    depth_avg = (abs(ls_depth) + abs(rs_depth)) / 2
    depth_diff_ratio = abs(ls_depth - rs_depth) / depth_avg if depth_avg else 1.0

    left_span = head_idx - ls_idx
    right_span = rs_idx - head_idx
    time_ratio = max(left_span, right_span) / min(left_span, right_span) if min(left_span, right_span) > 0 else 99

    neckline_slope_pct = abs(peak2_price - peak1_price) / peak1_price if peak1_price else 1.0

    pattern.quality_score = compute_quality_score(
        depth_diff_ratio, time_ratio, head_prom_pct, neckline_slope_pct,
        head_decline_vol, ls_decline_vol
    )

    # Scan forward from right shoulder for a neckline close-through with volume spike.
    # Shift by 1 so the comparison baseline EXCLUDES the breakout day itself —
    # otherwise a volume spike inflates its own threshold.
    avg20 = df['Volume'].rolling(20).mean().shift(1)

    breakout_idx = None
    for i in range(rs_idx + 1, len(df)):
        close = df['Close'].iloc[i]
        vol = df['Volume'].iloc[i]
        avg_vol = avg20.iloc[i]
        crossed = (close > pattern.neckline_price) if is_inverse else (close < pattern.neckline_price)
        if crossed:
            vol_spike = (not np.isnan(avg_vol)) and (vol >= avg_vol * VOLUME_BREAKOUT_MULT)
            breakout_idx = i
            pattern.breakout_date = pd.Timestamp(df['Date'].iloc[i])
            pattern.breakout_price = close
            pattern.breakout_volume_ratio = round(vol / avg_vol, 2) if (avg_vol and not np.isnan(avg_vol)) else None
            pattern.volume_confirmed = bool(vol_spike and head_volume_ok)
            break

    # --- Pre-breakout invalidation check ---
    # Murphy's structure requires price to stay beyond the right shoulder until the
    # neckline breaks. If price closes back past the right shoulder level BEFORE any
    # neckline break occurs, the setup is invalidated — a violated right shoulder means
    # this is no longer a valid H&S structure, even if price later happens to cross the
    # neckline anyway (that would be a different, fresh setup, not this one).
    search_end = breakout_idx if breakout_idx is not None else len(df)
    invalidation_date = None
    for j in range(rs_idx + 1, search_end):
        c = df['Close'].iloc[j]
        if is_inverse and c < pattern.right_shoulder_price:
            invalidation_date = df['Date'].iloc[j]
            break
        if not is_inverse and c > pattern.right_shoulder_price:
            invalidation_date = df['Date'].iloc[j]
            break

    if invalidation_date is not None:
        pattern.status = 'Invalidated'
        pattern.notes = (f"Price closed beyond right shoulder level on "
                          f"{pd.Timestamp(invalidation_date).date()} before neckline break — setup invalidated")
        # Clear any breakout that happened AFTER the invalidation — it doesn't validate
        # this original structure since the right shoulder was already broken by then.
        pattern.breakout_date = None
        pattern.breakout_price = None
        pattern.breakout_volume_ratio = None
        pattern.volume_confirmed = False
        return pattern

    if breakout_idx is not None and pattern.volume_confirmed:
        pattern.status = 'Confirmed'
        height = abs(pattern.neckline_price - pattern.head_price)
        if is_inverse:
            pattern.target = pattern.breakout_price + height
            pattern.stop_loss = pattern.right_shoulder_price
        else:
            pattern.target = pattern.breakout_price - height
            pattern.stop_loss = pattern.right_shoulder_price

        risk = abs(pattern.breakout_price - pattern.stop_loss)
        reward = abs(pattern.target - pattern.breakout_price)
        pattern.risk_reward = round(reward / risk, 2) if risk > 0 else None

        # check invalidation AND target-hit after breakout
        post_breakout = df.iloc[breakout_idx + 1:]
        if is_inverse:
            failed = (post_breakout['Close'] < pattern.stop_loss).any()
            pattern.target_hit = bool((post_breakout['Close'] >= pattern.target).any())
        else:
            failed = (post_breakout['Close'] > pattern.stop_loss).any()
            pattern.target_hit = bool((post_breakout['Close'] <= pattern.target).any())
        if failed:
            pattern.status = 'Failed'
            pattern.notes = "Closed beyond right-shoulder level after breakout"
    elif breakout_idx is not None and not pattern.volume_confirmed:
        pattern.status = 'False Start'
        pattern.notes = "Neckline crossed but volume/head-leg criteria not met — not confirmed"
    else:
        pattern.status = 'Forming'
        pattern.notes = "Neckline not yet broken"
        latest_avg_vol = df['Volume'].tail(20).mean()
        needed_vol = latest_avg_vol * VOLUME_BREAKOUT_MULT
        direction = "above" if is_inverse else "below"
        pattern.trigger_condition = (
            f"Confirms if daily close goes {direction} {pattern.neckline_price:.2f} "
            f"with volume >= {needed_vol:,.0f} shares (~{VOLUME_BREAKOUT_MULT}x 20-day avg)"
        )

    return pattern


# ----------------------------------------------------------------------
# Provisional (early, unconfirmed) right-shoulder detection
# ----------------------------------------------------------------------
def find_provisional_candidates(df: pd.DataFrame, symbol: str) -> List[PatternResult]:
    """
    Uses a relaxed pivot window to surface a potential right shoulder in the most
    recent bars — ones too close to today to have passed the full PIVOT_WINDOW=5
    fractal confirmation yet. These are explicitly flagged 'Provisional': the
    geometry looks plausible, but the low could still be undone by tomorrow's candle.
    A seasoned analyst would want visibility into this without mistaking it for a
    confirmed signal — hence the separate, clearly-labeled category downstream.
    """
    prov_pivots = find_pivots(df, PROVISIONAL_PIVOT_WINDOW)
    raw = find_inverse_hs_candidates(prov_pivots, symbol) + find_hs_candidates(prov_pivots, symbol)

    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(df['Date'])}
    tail_cutoff = len(df) - PIVOT_WINDOW  # bars from here on aren't yet confirmable by the standard detector

    provisional = []
    for r in raw:
        rs_idx = date_to_idx[r.right_shoulder_date]
        if rs_idx >= tail_cutoff:
            r.rs_confirmation = 'Provisional'
            provisional.append(r)
    return provisional


# ----------------------------------------------------------------------
# Watchlist classification
# ----------------------------------------------------------------------
def classify_for_watchlist(pattern: PatternResult, date_to_idx: dict, last_idx: int) -> Optional[str]:
    """
    Assigns a watchlist category for actionable, forward-looking setups only.
    Returns None for patterns that should NOT appear on a "what's about to happen"
    watchlist — i.e. Failed, Invalidated, False Start, or Confirmed patterns that are
    stale (breakout too long ago, or target already hit).

    Forming patterns get an ADAPTIVE staleness check: if the time waited for a
    breakout exceeds STALENESS_MULTIPLIER x the pattern's own average leg duration,
    it's flagged overdue rather than silently kept as a fresh "Watching" pick. A
    pattern that took 6 months to form gets more patience than one that took 3 weeks.
    """
    if pattern.status == 'Forming':
        ls_idx = date_to_idx[pattern.left_shoulder_date]
        head_idx = date_to_idx[pattern.head_date]
        rs_idx = date_to_idx[pattern.right_shoulder_date]
        avg_leg = ((head_idx - ls_idx) + (rs_idx - head_idx)) / 2
        rs_age = last_idx - rs_idx
        pattern.rs_age_days = rs_age
        pattern.is_stale = rs_age > (avg_leg * STALENESS_MULTIPLIER)

        rs_label = "Provisional RS" if pattern.rs_confirmation == 'Provisional' else "Confirmed RS"
        if pattern.is_stale:
            return f"Stale - {rs_label} (overdue, {rs_age}d since RS)"
        return f"Watching - {rs_label}"

    if pattern.status == 'Confirmed':
        breakout_idx = date_to_idx.get(pattern.breakout_date)
        if breakout_idx is None:
            return None
        days_since_breakout = last_idx - breakout_idx
        if days_since_breakout <= RECENT_BREAKOUT_DAYS and not pattern.target_hit:
            return 'Recent Breakout'
        return None

    return None  # Failed, False Start, Invalidated


# ----------------------------------------------------------------------
# Main scan function for a single stock's DataFrame
# ----------------------------------------------------------------------
def scan_symbol(df: pd.DataFrame, symbol: str) -> List[PatternResult]:
    df = df.reset_index(drop=True).sort_values('Date').reset_index(drop=True)
    if len(df) < (PIVOT_WINDOW * 2 + 20):
        return []

    pivots = find_pivots(df, PIVOT_WINDOW)
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(df['Date'])}

    # Confirmed-pivot candidates take priority; provisional ones only fill in gaps
    # (e.g. a very recent right shoulder the standard detector can't see yet).
    confirmed_candidates = find_inverse_hs_candidates(pivots, symbol) + find_hs_candidates(pivots, symbol)
    provisional_candidates = find_provisional_candidates(df, symbol)
    candidates = confirmed_candidates + provisional_candidates

    results = []
    for c in candidates:
        evaluated = evaluate_pattern(df, c, date_to_idx)
        results.append(evaluated)

    # Deduplicate: keep only the first candidate per (pattern_type, right_shoulder_date) —
    # confirmed_candidates were appended first, so they win any tie over provisional ones.
    seen = {}
    for r in results:
        key = (r.pattern_type, r.right_shoulder_date)
        if key not in seen:
            seen[key] = r

    final_results = list(seen.values())
    last_idx = len(df) - 1
    for r in final_results:
        r.watchlist_category = classify_for_watchlist(r, date_to_idx, last_idx)

    return final_results
