#!/usr/bin/env bash
# Cron entry point for Kalshi paper trading. Usage: kalshi_paper.sh morning|settle|decide
#   morning: scrape last night's box scores / injuries / rosters, then settle (laptop: scraping needs a home IP)
#   settle:  settle paper bets + report, no scrape (VPS)
#   decide:  score open points ladders and record taker + maker paper bets
set -uo pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.." || exit 1
source env/bin/activate
log="logs/kalshi_paper_$(date +%F).log"
echo "===== $(date '+%F %T') $1 =====" >> "$log"
case "$1" in
    morning)
        python main.py --scrape >> "$log" 2>&1
        python main.py --kalshi-paper-settle >> "$log" 2>&1
        python main.py --kalshi-paper-report >> "$log" 2>&1
        ;;
    settle)
        python main.py --kalshi-paper-settle >> "$log" 2>&1
        python main.py --kalshi-paper-report >> "$log" 2>&1
        ;;
    decide)
        python main.py --kalshi-paper-decide >> "$log" 2>&1
        ;;
    *)
        echo "usage: $0 morning|settle|decide" >&2; exit 2 ;;
esac
