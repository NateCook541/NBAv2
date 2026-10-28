#!/usr/bin/env bash
# Laptop side (basketball-reference blocks cloud IPs, so scraping can't run on the VPS).
# Cron runs this hourly; it scrapes + pushes once per day, the first hour the laptop is awake,
# so a closed laptop just catches up later instead of missing the day.
#   scrape_push.sh          scrape (once a day) then push
#   scrape_push.sh push     push only
# VPS host comes from $NBA_VPS (default: the ssh alias "nba-vps" in ~/.ssh/config)
set -uo pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.." || exit 1
source env/bin/activate
VPS="${NBA_VPS:-nba-vps}"
mkdir -p logs
log="logs/scrape_push_$(date +%F).log"
scraped="logs/.scraped_$(date +%F)"
pushed="logs/.pushed_$(date +%F)"

push() {
    # Never ship Kalshi tables: on the VPS they'd shadow the live kalshi.db
    if sqlite3 NBA.db "SELECT name FROM sqlite_master WHERE name LIKE 'Kalshi%'" | grep -q .; then
        echo "NBA.db still has Kalshi tables, run: python main.py --split-kalshi-db" >&2; return 1
    fi
    # And never push to a VPS that hasn't split yet, it would overwrite the paper bets
    if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$VPS" "test -f ~/NBAv2/kalshi.db"; then
        echo "VPS unreachable or not split yet (no ~/NBAv2/kalshi.db), not pushing" >&2; return 1
    fi
    # Consistent snapshot even if a scrape is mid-write, then an atomic swap on the VPS
    rm -f NBA.db.push
    sqlite3 NBA.db "VACUUM INTO 'NBA.db.push'" || return 1
    scp -q -o BatchMode=yes NBA.db.push "$VPS:NBAv2/NBA.db.incoming" \
        && ssh -o BatchMode=yes "$VPS" "mv ~/NBAv2/NBA.db.incoming ~/NBAv2/NBA.db" \
        && echo "pushed NBA.db to $VPS ($(du -h NBA.db.push | cut -f1))"
    local ok=$?
    rm -f NBA.db.push
    return $ok
}

# Done for today, stay quiet for the rest of the hourly runs
[ "${1:-}" != "push" ] && [ -e "$pushed" ] && exit 0

{
    echo "===== $(date '+%F %T') ${1:-scrape} ====="
    if [ "${1:-}" = "push" ]; then
        push
    else
        if [ ! -e "$scraped" ]; then
            if python main.py --scrape; then touch "$scraped"; else echo "scrape failed, will retry next hour"; fi
        fi
        # Push retries every hour until it lands (VPS down, no wifi, ...)
        [ -e "$scraped" ] && push && touch "$pushed"
    fi
} >> "$log" 2>&1
