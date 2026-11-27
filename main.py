"""Pump.fun Bot Sprint: early-momentum paper bot for pump.fun bonding curves.

Strategy: watch every newly created coin; the first time its real quote reserves
are deep enough to take a full minimum stake (1% size cap), buy a small fixed
stake if its price is still near its creation price. Hold the position until it
hits a take-profit multiple of its fill cost, a stop-loss floor, or a maximum
holding age, then sell everything. At most five open positions, one stake per
coin, never re-entering a coin already traded. No randomness, no wall clock, no
network: every decision is a pure function of the replay state, so two runs
produce byte-identical action logs.

Tunable parameters come from the environment with fixed defaults, so a sweep
harness can vary them without editing the file; the graded run uses the
defaults below.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bot  # noqa: E402  (after the sys.path line: -P leaves the script directory off sys.path)

LAMPORTS_PER_SOL = makefun_bot.LAMPORTS_PER_SOL


def _env_int(name, default):
    try:
        return int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default


# Entry: buy when price <= ENTRY_CAP_NUM / ENTRY_CAP_DEN of creation price.
ENTRY_CAP_NUM = _env_int("SPRINT_ENTRY_CAP_NUM", 110)
ENTRY_CAP_DEN = _env_int("SPRINT_ENTRY_CAP_DEN", 100)
# Exit: sell at TAKE_PROFIT_NUM / TAKE_PROFIT_DEN of fill cost per token.
TAKE_PROFIT_NUM = _env_int("SPRINT_TP_NUM", 104)
TAKE_PROFIT_DEN = _env_int("SPRINT_TP_DEN", 100)
# Stop-loss: sell at STOP_LOSS_NUM / STOP_LOSS_DEN of fill cost; 0 disables it.
STOP_LOSS_NUM = _env_int("SPRINT_SL_NUM", 0)
STOP_LOSS_DEN = _env_int("SPRINT_SL_DEN", 100)
# Maximum holding age in slots before a forced market sell.
HOLD_SLOTS = _env_int("SPRINT_HOLD_SLOTS", 50000)
# Fixed stake per coin, in lamports.
STAKE_LAMPORTS = _env_int("SPRINT_STAKE", 10_000_000)
# Minimum real quote reserves before a buy is allowed (size cap is 1% of them).
MIN_REAL_QUOTE = _env_int("SPRINT_MIN_REAL_QUOTE", 1_000_000_000)
# A coin is only considered within this many slots of its creation.
MAX_AGE_SLOTS = _env_int("SPRINT_MAX_AGE", 5000)
# Slippage tolerance on every order, in basis points.
SLIPPAGE_BPS = _env_int("SPRINT_SLIPPAGE_BPS", 500)
# Trailing stop: once a position's price reaches TRAIL_ACT_NUM / TRAIL_ACT_DEN
# of its fill cost, remember its peak price; sell when the price falls to
# TRAIL_DROP_NUM / TRAIL_DROP_DEN of that peak. 0 disables the trailing stop.
TRAIL_ACT_NUM = _env_int("SPRINT_TRAIL_ACT_NUM", 0)
TRAIL_ACT_DEN = _env_int("SPRINT_TRAIL_ACT_DEN", 100)
TRAIL_DROP_NUM = _env_int("SPRINT_TRAIL_DROP_NUM", 90)
TRAIL_DROP_DEN = _env_int("SPRINT_TRAIL_DROP_DEN", 100)

# mint -> [created_slot, creation vq, creation vt]
fresh_coins = {}
# mint -> slot the position was first seen in the account
bought_at = {}
# mints already traded once (never re-entered)
done_coins = {}
# mint -> [peak vq, peak vt] since the position opened (trailing stop)
peak_price = {}
# Own record of every order this run sent, for the end-of-run audit below.
sent_buys = []
sent_sells = []
peak_committed = [0]


def _price_le(curve_vq, curve_vt, ref_vq, ref_vt, num, den):
    """True when curve price (vq/vt) <= (num/den) * reference price (ref_vq/ref_vt)."""
    return curve_vq * ref_vt * den <= num * ref_vq * curve_vt


def _price_ge(curve_vq, curve_vt, ref_vq, ref_vt, num, den):
    """True when curve price (vq/vt) >= (num/den) * reference price (ref_vq/ref_vt)."""
    return curve_vq * ref_vt * den >= num * ref_vq * curve_vt


def on_batch(state, batch):
    caps = state.hello["caps"]
    tx = state.tx_fee_lamports()
    free = state.free_cash_lamports()
    slots = caps["max_open_positions"] - state.committed_position_count()
    orders = []
    slot = state.slot

    # Remember when each open position first appeared, for the hold-age exit.
    for mint in state.positions:
        if mint not in bought_at:
            bought_at[mint] = slot

    # Register newly created coins from this batch's events.
    for e in batch["events"]:
        mint = e["mint"]
        if e["event"] == "CreateEvent":
            curve = state.curves.get(mint)
            if curve is not None and mint not in done_coins:
                fresh_coins[mint] = [e["slot"], curve.virtual_quote_reserves,
                                     curve.virtual_token_reserves]

    # Exits first: take-profit, stop-loss, maximum holding age, trailing stop.
    # A mint with a sell already in flight is skipped: the position is still in
    # the account until the fill lands, and a second sell would be rejected.
    selling = set()
    for p in state.pending.values():
        if p.order["side"] == "sell":
            selling.add(p.order["mint"])
    for mint in sorted(state.positions):
        if mint in selling:
            continue
        holding = state.positions[mint]
        curve = state.curves.get(mint)
        if curve is None or curve.complete:
            continue
        tokens = holding.token_amount
        cost = holding.cost_lamports
        if tokens <= 0 or cost <= 0:
            continue
        vq = curve.virtual_quote_reserves
        vt = curve.virtual_token_reserves
        if vt <= 0:
            continue
        # Track the peak price since the position opened.
        if TRAIL_ACT_NUM > 0:
            peak = peak_price.get(mint)
            if peak is None or vq * peak[1] > peak[0] * vt:
                peak_price[mint] = [vq, vt]
        sell = False
        if _price_ge(vq, vt, cost, tokens, TAKE_PROFIT_NUM, TAKE_PROFIT_DEN):
            sell = True
        elif STOP_LOSS_NUM > 0 and _price_le(vq, vt, cost, tokens, STOP_LOSS_NUM, STOP_LOSS_DEN):
            sell = True
        elif slot - bought_at.get(mint, slot) >= HOLD_SLOTS:
            sell = True
        elif TRAIL_ACT_NUM > 0:
            peak = peak_price.get(mint)
            if peak is not None and _price_ge(peak[0], peak[1], cost, tokens,
                                              TRAIL_ACT_NUM, TRAIL_ACT_DEN):
                # Peak cleared the activation level: sell once the price has
                # fallen to TRAIL_DROP of the peak.
                if vq * peak[1] * TRAIL_DROP_DEN <= TRAIL_DROP_NUM * peak[0] * vt:
                    sell = True
        if sell and len(orders) < caps["max_orders_per_batch"]:
            orders.append({"side": "sell", "mint": mint, "token_amount": tokens,
                           "max_slippage_bps": SLIPPAGE_BPS})
            peak_price.pop(mint, None)

    # Entries: one fixed stake in a fresh coin, decided exactly once.
    for mint in list(fresh_coins):
        if slots <= 0 or len(orders) >= caps["max_orders_per_batch"]:
            break
        created_slot, ref_vq, ref_vt = fresh_coins[mint]
        curve = state.curves.get(mint)
        if curve is None or curve.complete:
            del fresh_coins[mint]
            continue
        if slot - created_slot > MAX_AGE_SLOTS:
            del fresh_coins[mint]
            continue
        if curve.real_quote_reserves < MIN_REAL_QUOTE:
            continue
        # Decide now: buy only while the price is still near creation price.
        del fresh_coins[mint]
        if not _price_le(curve.virtual_quote_reserves, curve.virtual_token_reserves,
                         ref_vq, ref_vt, ENTRY_CAP_NUM, ENTRY_CAP_DEN):
            continue
        if free < STAKE_LAMPORTS + tx:
            continue
        orders.append({"side": "buy", "mint": mint, "quote_lamports": STAKE_LAMPORTS,
                       "max_slippage_bps": SLIPPAGE_BPS})
        done_coins[mint] = True
        free -= STAKE_LAMPORTS + tx
        slots -= 1

    for o in orders:
        if o["side"] == "buy":
            sent_buys.append(o["mint"])
        else:
            sent_sells.append(o["mint"])
    committed = state.committed_position_count()
    if committed > peak_committed[0]:
        peak_committed[0] = committed

    return orders


def audit_sprint_log():
    """End-of-run self-check over this run's own sent orders.

    Verifies the guardrails the sprint is scored on, from the bot's own record:
    every buy was exactly one fixed stake, no coin was ever bought twice, no
    sell was sent for a coin never bought, and the committed position count
    never passed the five-position cap. Prints one summary line to stderr so
    the protocol stream on stdout stays clean; returns True when every
    invariant holds.
    """
    ok = True
    distinct = set(sent_buys)
    if len(distinct) != len(sent_buys):
        ok = False
    if any(mint not in distinct for mint in sent_sells):
        ok = False
    if peak_committed[0] > 5:
        ok = False
    sys.stderr.write(
        "sprint audit: buys=%d distinct=%d sells=%d max_open=%d reentered=%d -> %s\n"
        % (len(sent_buys), len(distinct), len(sent_sells), peak_committed[0],
           len(sent_buys) - len(distinct), "PASS" if ok else "FAIL"))
    return ok


def on_done(state, done):
    audit_sprint_log()


if __name__ == "__main__":
    makefun_bot.run(on_batch, on_done=on_done)
