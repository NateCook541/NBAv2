"""
Real Kalshi orders for the paper trader's picks (small stakes, taker only).

Runs right after each paper decide. Every live order is the twin of a paper bet (KalshiPaperBets.bet_id),
so live fills, fees and CLV can be compared with paper on the exact same bets:

  1. Pick: this run's paper taker bets on the allowed sides (NO only for now, YES stays paper)
  2. Guards: master switch, kill-switch file, stale box scores, nightly risk cap, account balance
  3. Re-quote: fetch the market again, buy at the current price only if the edge still clears the bar
     and the price is no more than LIVE_MAX_SLIPPAGE worse than the paper price
  4. Order: limit at that price, immediate_or_cancel, so it can never rest into the game
  5. Reconcile: read the fills back from Kalshi (count, price, fees)
  6. Settle: grade filled orders once the market settles (settlePaperBets calls settleLiveOrders)

Modes: with LIVE_TRADING off or --kalshi-dry-run, orders are built and signed but not sent, and stored as
mode "dry" (authenticated reads like the balance still run, so a dry run also checks the API key works).
The live row is written before the order is sent, so a crash can never send the same order twice:
the next run finds the row and reconciles it instead.

Credentials (never committed): .env in the repo root with
  KALSHI_API_KEY_ID=<key id>
  KALSHI_PRIVATE_KEY_PATH=/home/nate/.kalshi/kalshi_key.pem   (chmod 600)
"""

import base64
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from betting.backtest import _kalshiFee, _sidePayout
from betting.kalshiCollector import BASE_URL, _fmtUtc, _throttle, _toFloat
from config import (
    LIVE_KILL_FILE, LIVE_MAX_BET_AGE_MIN, LIVE_MAX_NIGHT_RISK, LIVE_MAX_SLIPPAGE, LIVE_MIN_EV, LIVE_MIN_TIP_MIN,
    LIVE_ORDER_TYPES, LIVE_PRICE_BAND, LIVE_SIDES, LIVE_STAKE, LIVE_TRADING,
)
from data.dbManager import DBManager, connect

PT = ZoneInfo("America/Los_Angeles")
API_PREFIX = "/trade-api/v2"


# AUTH + HTTP


class KalshiAuth:
    """RSA-PSS request signing: sign(timestamp_ms + METHOD + path without query) with the account's private key"""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.keyID = os.environ.get("KALSHI_API_KEY_ID")
        keyPath = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        self.key = None
        if self.keyID and keyPath and Path(keyPath).expanduser().exists():
            from cryptography.hazmat.primitives import serialization
            self.key = serialization.load_pem_private_key(Path(keyPath).expanduser().read_bytes(), password=None)

    @property
    def ready(self):
        return self.key is not None

    def headers(self, method, path):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{API_PREFIX}{path.split('?')[0]}".encode()
        sig = self.key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                             salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.keyID, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "Content-Type": "application/json"}


def _request(auth, method, path, params=None, body=None, retries=4):
    """Signed request. POSTs are never retried (a retry could double an order), GETs back off on 429/5xx"""
    for attempt in range(retries if method == "GET" else 1):
        _throttle()
        resp = requests.request(method, f"{BASE_URL}{path}", params=params,
                                data=json.dumps(body) if body is not None else None,
                                headers=auth.headers(method, path), timeout=30)
        if method == "GET" and (resp.status_code == 429 or resp.status_code >= 500):
            time.sleep(2 ** attempt)
            continue
        if resp.status_code >= 400:
            raise requests.HTTPError(f"{resp.status_code} {method} {path}: {resp.text[:300]}", response=resp)
        return resp.json() if resp.text else {}
    resp.raise_for_status()


def _num(d, *keys, cents=False):
    """First present numeric field. Kalshi is moving to *_dollars / *_fp strings, older fields are cents / ints"""
    for k in keys:
        v = _toFloat(d.get(k)) if d else None
        if v is not None:
            return v / 100 if cents and not k.endswith("_dollars") else v
    return None


def getBalance(auth):
    data = _request(auth, "GET", "/portfolio/balance")
    return _num(data, "balance_dollars", "balance", cents=True)


def _getQuote(ticker):
    """Public market read: current yes bid/ask (+ sizes) and status"""
    from betting.kalshiCollector import _get
    m = _get(f"/markets/{ticker}").get("market", {})
    return {"status": m.get("status"), "yes_bid": _toFloat(m.get("yes_bid_dollars")),
            "yes_ask": _toFloat(m.get("yes_ask_dollars")), "yes_bid_size": _toFloat(m.get("yes_bid_size_fp")),
            "yes_ask_size": _toFloat(m.get("yes_ask_size_fp"))}


def _fillsFor(auth, orderID):
    fills, cursor = [], None
    while True:
        params = {"order_id": orderID, "limit": 200, **({"cursor": cursor} if cursor else {})}
        data = _request(auth, "GET", "/portfolio/fills", params=params)
        fills.extend(data.get("fills", []))
        cursor = data.get("cursor")
        if not cursor:
            return fills


def _estFee(price, count):
    """Kalshi's taker fee: 0.07 * C * P * (1 - P), rounded up to the cent per order"""
    return math.ceil(round(_kalshiFee(price) * count * 100, 6)) / 100 if count else 0.0


def _summarizeFills(auth, order, side, limitPrice):
    """Filled count, average price, cost and fees for an order, from the fills endpoint (order fields as fallback)"""
    orderID = order.get("order_id")
    fills = _fillsFor(auth, orderID) if orderID else []
    count = sum(_num(f, "count_fp", "count") or 0 for f in fills)
    cost = sum((_num(f, "count_fp", "count") or 0)
               * (_num(f, f"{side}_price_dollars", f"{side}_price", cents=True) or limitPrice) for f in fills)
    fees = [_num(f, "fee_cost_dollars", "fee_cost") for f in fills]
    if fills and all(v is not None for v in fees):
        fee, feeSource = sum(fees), "fills"
    else:
        fee = _num(order, "taker_fees_dollars", "taker_fees", cents=True)
        feeSource = "order"
        if fee is None:
            fee, feeSource = _estFee(limitPrice, count), "estimated"
    if not fills:
        # Fall back to the order's own counters
        count = _num(order, "fill_count_fp", "fill_count", "taker_fill_count") or 0
        cost = _num(order, "taker_fill_cost_dollars", "taker_fill_cost", cents=True) or count * limitPrice
    return {"filled_count": count, "avg_price": cost / count if count else None, "cost": round(cost, 4),
            "fees": round(fee or 0.0, 4), "fee_source": feeSource}


# GUARDS


def _boxScoreLag(conn):
    """(newest box score date, newest past game date with Kalshi markets). Stale when the second is newer"""
    lastBox = conn.execute("""SELECT MAX(g.game_date) FROM Player_game_logs l JOIN Games g USING (game_id)""").fetchone()[0]
    today = datetime.now(PT).strftime("%Y-%m-%d")
    lastPlayed = conn.execute("SELECT MAX(game_date) FROM KalshiMarkets WHERE game_date < ?", (today,)).fetchone()[0]
    return lastBox, lastPlayed


def _nightRisk(conn, gameDate, mode):
    row = conn.execute(
        """SELECT COALESCE(SUM(CASE WHEN status IN ('executed', 'canceled') THEN cost + fees
                                    WHEN status IN ('planned', 'submitting') THEN count * limit_price ELSE 0 END), 0)
           FROM KalshiLiveOrders WHERE game_date = ? AND mode = ?""",
        (gameDate, mode),
    ).fetchone()
    return row[0] or 0.0


# PLACE


def placeLiveOrders(dbPath="NBA.db", dryRun=False):
    """Send this decide run's eligible paper picks to Kalshi (or sign + log them in dry mode)"""
    mode = "live" if (LIVE_TRADING and not dryRun) else "dry"
    tag = f"[live:{mode}]"
    auth = KalshiAuth()
    db = DBManager(dbPath)
    if mode == "live" and not auth.ready:
        print(f"{tag} no API key (KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH in .env), nothing sent")
        return []
    if Path(LIVE_KILL_FILE).exists():
        print(f"{tag} kill switch {LIVE_KILL_FILE} present, nothing sent")
        return []

    conn = connect(dbPath)
    _reconcilePending(conn, auth, db, mode, tag)

    lastBox, lastPlayed = _boxScoreLag(conn)
    if lastPlayed and (lastBox is None or lastBox < lastPlayed):
        print(f"{tag} box scores stale (through {lastBox}, games played through {lastPlayed}), no live orders")
        conn.close()
        return []

    now = datetime.now(timezone.utc)
    sides = ",".join(f"'{s}'" for s in LIVE_SIDES)
    types = ",".join(f"'{t}'" for t in LIVE_ORDER_TYPES)
    picks = pd.read_sql_query(
        f"""
        SELECT b.* FROM KalshiPaperBets b
        WHERE b.order_type IN ({types}) AND b.side IN ({sides})
          AND b.decision_ts BETWEEN ? AND ? AND b.est_tip_utc >= ?
          AND NOT EXISTS (SELECT 1 FROM KalshiLiveOrders o WHERE o.bet_id = b.bet_id AND o.mode = ?)
        ORDER BY b.ev DESC
        """,
        conn,
        params=[_fmtUtc(now - timedelta(minutes=LIVE_MAX_BET_AGE_MIN)), _fmtUtc(now),
                _fmtUtc(now + timedelta(minutes=LIVE_MIN_TIP_MIN)), mode],
    )
    if picks.empty:
        print(f"{tag} no new eligible picks (taker, sides {LIVE_SIDES})")
        conn.close()
        return []

    balance = None
    if auth.ready:
        try:
            balance = getBalance(auth)
            print(f"{tag} account balance ${balance:.2f}")
        except requests.HTTPError as e:
            print(f"{tag} balance check failed: {e}")
            if mode == "live":
                conn.close()
                return []
    else:
        print(f"{tag} no API key configured, dry run without auth checks")

    rows = []
    for b in picks.itertuples(index=False):
        row = {"bet_id": int(b.bet_id), "mode": mode, "client_order_id": f"nbav2-{int(b.bet_id)}",
               "game_date": b.game_date, "player_name": b.player_name, "market_ticker": b.market_ticker,
               "strike": b.strike, "side": b.side, "paper_price": b.price, "paper_ev": b.ev,
               "est_tip_utc": b.est_tip_utc, "quote_ts": _fmtUtc(datetime.now(timezone.utc))}
        try:
            q = _getQuote(b.market_ticker)
        except requests.HTTPError as e:
            rows.append(_skip(db, row, f"quote failed: {str(e)[:80]}"))
            continue
        row.update(quote_yes_bid=q["yes_bid"], quote_yes_ask=q["yes_ask"])
        if q["status"] not in ("active", "open") or q["yes_bid"] is None or q["yes_ask"] is None:
            rows.append(_skip(db, row, f"market {q['status']} / no two-sided quote"))
            continue
        # Buying NO lifts a YES bid: we pay 1 - yes_bid
        price = round(q["yes_ask"] if b.side == "yes" else 1 - q["yes_bid"], 4)
        pSide = b.p_bet if b.side == "yes" else 1 - b.p_bet
        ev = pSide - price - _kalshiFee(price)
        count = int(LIVE_STAKE // price) if price > 0 else 0
        row.update(limit_price=price, live_ev=round(ev, 4), count=count)
        if price > b.price + LIVE_MAX_SLIPPAGE + 1e-9:
            rows.append(_skip(db, row, f"price moved {b.price:.2f} -> {price:.2f}"))
            continue
        if ev < LIVE_MIN_EV or not (LIVE_PRICE_BAND[0] <= price <= LIVE_PRICE_BAND[1]):
            rows.append(_skip(db, row, f"edge gone at {price:.2f} (ev {ev:.3f})"))
            continue
        if count < 1:
            rows.append(_skip(db, row, "stake below one contract"))
            continue
        risk = count * price + _estFee(price, count)
        if _nightRisk(conn, b.game_date, mode) + risk > LIVE_MAX_NIGHT_RISK:
            rows.append(_skip(db, row, f"nightly cap ${LIVE_MAX_NIGHT_RISK:.0f} reached"))
            continue
        if balance is not None and risk > balance:
            rows.append(_skip(db, row, f"balance ${balance:.2f} too low"))
            continue

        body = {"ticker": b.market_ticker, "client_order_id": row["client_order_id"], "action": "buy",
                "side": b.side, "count": count, "type": "limit",
                f"{b.side}_price": int(round(price * 100)), "time_in_force": "immediate_or_cancel"}
        row["request_json"] = json.dumps(body)
        if mode == "dry":
            if auth.ready:
                auth.headers("POST", "/portfolio/orders")   # proves the key can sign
            row["status"] = "planned"
            db.insertKalshiLiveOrder(row)
            rows.append(row)
            print(f"{tag} would buy {count} {b.side.upper()} {b.market_ticker} @ {price:.2f} "
                  f"(paper {b.price:.2f}, ev {ev:.3f})")
            continue

        # Row first, then the order: a crash after sending leaves a 'submitting' row that the next run
        # reconciles by client_order_id, never a second order
        row["status"] = "submitting"
        row["submitted_ts"] = _fmtUtc(datetime.now(timezone.utc))
        db.insertKalshiLiveOrder(row)
        try:
            resp = _request(auth, "POST", "/portfolio/orders", body=body)
        except requests.HTTPError as e:
            db.updateKalshiLiveOrder({"bet_id": row["bet_id"], "mode": mode, "status": "error", "error": str(e)[:500]})
            print(f"{tag} order failed {b.market_ticker}: {e}")
            continue
        order = resp.get("order", resp)
        upd = {"bet_id": row["bet_id"], "mode": mode, "order_id": order.get("order_id"),
               "response_json": json.dumps(order)[:4000], "status": order.get("status") or "executed",
               **_summarizeFills(auth, order, b.side, price)}
        db.updateKalshiLiveOrder(upd)
        balance = None if balance is None else balance - (upd["cost"] + upd["fees"])
        print(f"{tag} {b.side.upper()} {b.market_ticker}: filled {upd['filled_count']:.0f}/{count} "
              f"@ {upd['avg_price'] or 0:.3f} fees ${upd['fees']:.2f} ({upd['fee_source']})")
        rows.append({**row, **upd})
    conn.close()
    placed = [r for r in rows if r.get("status") not in ("skipped", None)]
    print(f"{tag} {len(placed)} orders {'sent' if mode == 'live' else 'planned'}, "
          f"{sum(r.get('status') == 'skipped' for r in rows)} skipped")
    return rows


def _skip(db, row, reason):
    row = {**row, "status": "skipped", "skip_reason": reason}
    db.insertKalshiLiveOrder(row)
    print(f"[live:{row['mode']}] skip {row['market_ticker']}: {reason}")
    return row


def _reconcilePending(conn, auth, db, mode, tag):
    """Rows left 'submitting' by a crash: find the order by client_order_id and record what filled"""
    if mode != "live":
        return
    stuck = pd.read_sql_query("SELECT * FROM KalshiLiveOrders WHERE mode = 'live' AND status = 'submitting'", conn)
    for r in stuck.itertuples(index=False):
        try:
            orders = _request(auth, "GET", "/portfolio/orders", params={"ticker": r.market_ticker}).get("orders", [])
        except requests.HTTPError as e:
            print(f"{tag} reconcile failed for {r.client_order_id}: {e}")
            continue
        order = next((o for o in orders if o.get("client_order_id") == r.client_order_id), None)
        if order is None:
            db.updateKalshiLiveOrder({"bet_id": r.bet_id, "mode": "live", "status": "error",
                                      "error": "not found on Kalshi after crash (never sent)"})
            continue
        upd = {"bet_id": r.bet_id, "mode": "live", "order_id": order.get("order_id"),
               "response_json": json.dumps(order)[:4000], "status": order.get("status") or "executed",
               **_summarizeFills(auth, order, r.side, r.limit_price)}
        db.updateKalshiLiveOrder(upd)
        print(f"{tag} reconciled {r.client_order_id}: filled {upd['filled_count']:.0f}")


# SETTLE


def settleLiveOrders(dbPath="NBA.db", dryRun=False):
    """Grade orders whose market has settled. Dry rows are graded as if fully filled at the limit (plumbing check)"""
    conn = connect(dbPath)
    rows = pd.read_sql_query(
        """
        SELECT o.*, m.result AS mkt_result, m.settlement_value, b.close_bid AS twin_close_bid,
               b.close_ask AS twin_close_ask
        FROM KalshiLiveOrders o
        JOIN KalshiMarkets m ON m.market_ticker = o.market_ticker
        LEFT JOIN KalshiPaperBets b ON b.bet_id = o.bet_id
        WHERE o.pnl IS NULL AND o.status IN ('executed', 'canceled', 'planned') AND m.result IS NOT NULL
        """,
        conn,
    )
    conn.close()
    updates = []
    for r in rows.itertuples(index=False):
        if r.status == "planned":
            count, cost, fees, avg = r.count, r.count * r.limit_price, _estFee(r.limit_price, r.count), r.limit_price
        else:
            count, cost, fees, avg = r.filled_count or 0, r.cost or 0.0, r.fees or 0.0, r.avg_price
        payout = count * _sidePayout(r.side, r.mkt_result, r.settlement_value)
        upd = {"bet_id": r.bet_id, "mode": r.mode, "result": r.mkt_result, "payout": round(payout, 4),
               "pnl": round(payout - cost - fees, 4)}
        if avg is not None and pd.notna(r.twin_close_bid) and pd.notna(r.twin_close_ask):
            mid = (r.twin_close_bid + r.twin_close_ask) / 2
            upd["clv"] = round((mid if r.side == "yes" else 1 - mid) - avg, 4)
        updates.append(upd)
    if updates and not dryRun:
        db = DBManager(dbPath)
        for u in updates:
            db.updateKalshiLiveOrder(u)
    print(f"[live] settled {len(updates)} orders")
    return updates


# REPORT


def liveReport(dbPath="NBA.db", startDate=None, endDate=None):
    conn = connect(dbPath)
    o = pd.read_sql_query(
        """
        SELECT o.*, b.price AS twin_price, b.fee AS twin_fee, b.pnl AS twin_pnl, b.stake AS twin_stake,
               b.clv AS twin_clv
        FROM KalshiLiveOrders o LEFT JOIN KalshiPaperBets b ON b.bet_id = o.bet_id
        WHERE o.game_date >= ? AND o.game_date <= ?
        """,
        conn, params=[startDate or "0000", endDate or "9999"],
    )
    conn.close()
    if o.empty:
        return
    print("\nLive orders (real Kalshi orders on the paper picks; 'dry' = signed, not sent)")
    for mode, m in o.groupby("mode"):
        skipped = m[m["status"] == "skipped"]
        sent = m[m["status"].isin(["executed", "canceled", "planned"])]
        print(f"  {mode}: {len(m)} picks | {len(sent)} {'sent' if mode == 'live' else 'planned'} | "
              f"{len(skipped)} skipped | {(m['status'] == 'error').sum()} errors")
        if not skipped.empty:
            reasons = skipped["skip_reason"].str.split(r" [\d$(]", n=1, regex=True).str[0].value_counts()
            print("    skip reasons: " + ", ".join(f"{k} {v}" for k, v in reasons.items()))
        if mode == "live" and not sent.empty:
            want = sent["count"].sum()
            got = sent["filled_count"].fillna(0).sum()
            f = sent[sent["filled_count"].fillna(0) > 0]
            slip = (f["avg_price"] - f["twin_price"]) * 100
            feeModel = (f["twin_fee"] * f["filled_count"]).sum()
            print(f"    fill rate {got / want:.0%} of contracts | {len(f)} orders with fills | "
                  f"price vs paper {slip.mean():+.2f}c | fees ${f['fees'].sum():.2f} vs modeled ${feeModel:.2f}")
        g = sent[sent["pnl"].notna()]
        if not g.empty:
            staked = pd.to_numeric(g["cost"], errors="coerce").fillna(g["count"] * g["limit_price"]).sum()
            twinRoi = (g["twin_pnl"] / g["twin_stake"]).mean()
            print(f"    graded {len(g)} | PnL ${g['pnl'].sum():+.2f} on ${staked:.2f} "
                  f"({g['pnl'].sum() / staked:+.1%}) | CLV {g['clv'].mean() * 100:+.2f}c "
                  f"vs paper twins {g['twin_clv'].mean() * 100:+.2f}c | twin ROI {twinRoi:+.1%}")
