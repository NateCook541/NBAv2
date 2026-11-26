#!/usr/bin/env bash
# Cron entry point for Kalshi paper trading. Usage: kalshi_paper.sh morning|settle|decide
#   morning: scrape last night's box scores / injuries / rosters, then settle (laptop: scraping needs a home IP)
#   settle:  settle paper bets + report, no scrape (VPS)
#   decide:  score open points ladders and record taker + maker paper bets
#   news:    poll the official injury report, record news trader shadow bets (cron every 5 min, afternoons)
set -uo pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.." || exit 1
source env/bin/activate
mkdir -p logs
log="logs/kalshi_paper_$(date +%F).log"
[ "$1" != news ] && echo "===== $(date '+%F %T') $1 =====" >> "$log"
# One job at a time (the VPS has ~1GB and the injury report parser starts a JVM):
# the news poll skips its turn if anything is running, the others wait for it
exec 9> logs/.kalshi.lock
if [ "$1" = news ]; then
    flock -n 9 || exit 0
else
    flock -w 900 9 || { echo "lock busy for 15 min, skipping $1" >> "$log"; exit 1; }
fi
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
        # Real orders on this run's picks (dry unless config.LIVE_TRADING is on)
        python main.py --kalshi-live >> "$log" 2>&1
        ;;
    news)
        # Own log, it runs every few minutes
        python main.py --kalshi-news >> "logs/kalshi_news_$(date +%F).log" 2>&1
        ;;
    *)
        echo "usage: $0 morning|settle|decide|news" >&2; exit 2 ;;
esac
