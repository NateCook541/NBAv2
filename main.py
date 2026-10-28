import argparse
from config import CURRENT_SEASON
from pipeline.orchestrator import Pipeline

def main():
    parser = argparse.ArgumentParser(description="NBA prediction proj")
 
    # Scrape args
    parser.add_argument("--scrape", action="store_true")
    parser.add_argument("--backfill-from", type=str, default=None)
    parser.add_argument("--num-games",  type=int, default=None)
    parser.add_argument("--historical-seasons", type=int, nargs="+", default=None)
    # Per-game team ratings from the NBA stats API. Optional season(s); omit for all.
    # Must run from an IP the NBA API doesn't block.
    parser.add_argument("--scrape-ratings", nargs="*", type=int, metavar="SEASON", default=None)
    
    # Train args
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-end-date", type=str, default=None)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--cache-data", action="store_true")

    # Calibrator args
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--calibrator", action="store_true")
    parser.add_argument("--refit-calibrator", action="store_true")

    # Minutes args
    parser.add_argument("--train-minutes", action="store_true")

    # Results (game totals) model args
    parser.add_argument("--train-results", action="store_true")
    parser.add_argument("--cache-results-features", action="store_true")
    parser.add_argument("--eval-results-calibrator", action="store_true")
    parser.add_argument("--eval-results-layers", action="store_true",
                        help="Per-layer totals eval: MAE/bias sliced by line availability + calibration reliability")
    parser.add_argument("--line-having-split", action="store_true",
                        help="With --eval-results-layers: restrict to line-having games and split those, so the test fold contains the market line")

    # Props args
    parser.add_argument("--pull-props", nargs=2, metavar=("START_DATE", "END_DATE"))
    parser.add_argument("--pull-totals", nargs=2, metavar=("START_DATE", "END_DATE"),
                        help="Backfill game totals lines into Odds_archive for the "
                             "date window (inclusive). Pair with --totals-dry-run first.")
    parser.add_argument("--totals-dry-run", action="store_true",
                        help="Estimate --pull-totals credit cost from the local "
                             "schedule without spending any credits.")
    parser.add_argument("--totals-max-credits", type=int, default=None,
                        metavar="N",
                        help="Hard credit cap for --pull-totals. Stops before any "
                             "date whose full slate would exceed N, and prints where "
                             "to resume. Combine with --totals-dry-run to see how "
                             "many games N buys.")
    parser.add_argument("--verify-totals", nargs="?", const="__auto__", metavar="DATE",
                        help="Cheap end-to-end check (~11 credits): pull ONE game, "
                             "store it, and prove it joins to Games/Results the way "
                             "the backtest does. Optional DATE (YYYY-MM-DD), else "
                             "auto-picks the first post-archive game.")

    # Live / CLV validation args
    parser.add_argument("--freshness", nargs="?", const="__today__", metavar="DATE")
    parser.add_argument("--snapshot-open", action="store_true")
    parser.add_argument("--snapshot-close", action="store_true")
    parser.add_argument("--snapshot-dry-run", action="store_true")
    parser.add_argument("--score-live", nargs="?", const="__today__", metavar="DATE")
    parser.add_argument("--compute-clv", nargs="?", const="__today__", metavar="DATE")
    parser.add_argument("--clv-report", nargs="*", metavar=("START", "END"))

    # Kalshi prop collection (public market data, no key needed)
    parser.add_argument("--kalshi-snapshot", choices=["open", "pregame", "close"], default=None)
    parser.add_argument("--kalshi-window", type=int, default=None, metavar="MIN",
                        help="Only snapshot markets whose game tips within the next MIN minutes (close cron)")
    parser.add_argument("--kalshi-settle", nargs="?", const="__auto__", metavar="DATE",
                        help="Fill results for tracked markets through DATE (default yesterday ET)")
    parser.add_argument("--kalshi-backfill", nargs="*", metavar=("START", "END"),
                        help="Pull settled markets + hourly candles from Kalshi's archive (optional game date range)")
    parser.add_argument("--kalshi-series", nargs="+", default=["KXNBAPTS"],
                        help="Series to backfill (default points only)")
    parser.add_argument("--kalshi-dry-run", action="store_true")

    # Kalshi paper trading (points ladders)
    parser.add_argument("--kalshi-paper-decide", action="store_true",
                        help="Score open points ladders and record taker + maker paper bets")
    parser.add_argument("--kalshi-paper-lead", type=float, nargs=2, default=[1.0, 24.0], metavar=("MIN", "MAX"),
                        help="Only decide on games tipping between MIN and MAX hours from now")
    parser.add_argument("--kalshi-paper-settle", action="store_true",
                        help="Settle results, check maker fills and closing quotes for tipped games")
    parser.add_argument("--kalshi-paper-report", nargs="*", metavar=("START", "END"))
    parser.add_argument("--kalshi-fit-shrink", nargs="?", const="__paper__", metavar="RUNGS_CSV",
                        help="Fit the model-vs-market weight (default: settled paper rungs)")
    parser.add_argument("--kalshi-shrink-since", default=None, metavar="DATE")
    parser.add_argument("--split-kalshi-db", action="store_true",
                        help="One-time: move Kalshi tables out of the NBA db into kalshi.db next to it")

    # Backtest args
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--backtest-unders", action="store_true")
    parser.add_argument("--backtest-combined", action="store_true")
    parser.add_argument("--backtest-kalshi", action="store_true",
                        help="Walk-forward backtest vs archived Kalshi points ladders (uses --start-date/--end-date)")
    parser.add_argument("--kalshi-ev-thresh", type=float, default=0.03)
    parser.add_argument("--kalshi-price-band", type=float, nargs=2, default=[0.20, 0.80], metavar=("MIN", "MAX"))
    parser.add_argument("--kalshi-lead-hours", type=float, default=2,
                        help="Hours before tip the bet decision is made")
    parser.add_argument("--kalshi-no-shrink", action="store_true",
                        help="Bet on raw model probabilities instead of shrinking toward the market mid")
    parser.add_argument("--kalshi-max-spread", type=float, default=None,
                        help="Only bet rungs whose decision-time spread is at most this (e.g. 0.01)")
    parser.add_argument("--kalshi-reuse-rungs", action="store_true",
                        help="Skip retraining, re-pick bets from saved backtest_kalshi_rungs.csv")
    parser.add_argument("--over-edge-thresh", type=float, default=0.10)
    parser.add_argument("--under-edge-thresh", type=float, default=0.05)
    parser.add_argument("--edge-thresh", type=float, default=0.03)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--retrain-every-months", type=int, default=0)
    parser.add_argument("--retrain-minutes", action="store_true")
    parser.add_argument("--under-kelly-frac", type=float, default=None)
    parser.add_argument("--under-daily-cap", type=float, default=None)
    parser.add_argument("--under-max-stake", type=float, default=None)
    parser.add_argument("--holdout-train-end", type=str, default=None)
    
    # Backtest testing args
    parser.add_argument("--backtest-fold-test", action="store_true")

    # Shared args
    parser.add_argument("--db",  default="NBA.db")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    pipeline = Pipeline(dbPath=args.db)

    if args.quiet:
        from metrics.reporter import Reporter
        Reporter.verbose = False

    if args.train:
        pipeline.train(endDate=args.train_end_date, 
                       runMetrics=args.metrics,
        )

    if args.backtest:
        pipeline.backtest(
            startDate=args.start_date,
            endDate=args.end_date,
            edgeThresh=args.edge_thresh,
            bankroll=args.bankroll,
            retrainEveryMonths=(
                args.retrain_every_months
                if args.retrain_every_months > 0
                else None
            ),
            retrainMinutes=args.retrain_minutes,
        )
    
    if args.backtest_unders:
        kwargs = dict(
            startDate=args.start_date,
            endDate=args.end_date,
            bankroll=args.bankroll,
            retrainEveryMonths=(
                args.retrain_every_months
                if args.retrain_every_months > 0
                else 1
            ),
            retrainMinutes=args.retrain_minutes,
        )
        if args.under_kelly_frac is not None:
            kwargs["kellyFrac"] = args.under_kelly_frac
        if args.under_daily_cap is not None:
            kwargs["maxDailyExposure"] = args.under_daily_cap
        if args.under_max_stake is not None:
            kwargs["maxStakeAbs"] = args.under_max_stake
        pipeline.backtestUnders(**kwargs)

    if args.backtest_combined:
        kwargs = dict(
            startDate=args.start_date,
            endDate=args.end_date,
            bankroll=args.bankroll,
            overEdgeThresh=args.over_edge_thresh,
            underEdgeThresh=args.under_edge_thresh,
            retrainEveryMonths=(
                args.retrain_every_months
                if args.retrain_every_months > 0
                else 1
            ),
            retrainMinutes=args.retrain_minutes,
        )
        if args.under_kelly_frac is not None:
            kwargs["kellyFrac"] = args.under_kelly_frac
        if args.under_daily_cap is not None:
            kwargs["maxDailyExposure"] = args.under_daily_cap
        if args.under_max_stake is not None:
            kwargs["maxStakeAbs"] = args.under_max_stake
        if args.holdout_train_end is not None:
            kwargs["singleTrainEndDate"] = args.holdout_train_end
        pipeline.backtestCombined(**kwargs)

    if args.backtest_kalshi:
        pipeline.backtestKalshi(
            startDate=args.start_date, endDate=args.end_date,
            evThresh=args.kalshi_ev_thresh,
            minPrice=args.kalshi_price_band[0], maxPrice=args.kalshi_price_band[1],
            decisionLeadHours=args.kalshi_lead_hours,
            retrainEveryMonths=args.retrain_every_months or 1,
            shrink=not args.kalshi_no_shrink, maxSpread=args.kalshi_max_spread,
            reuseRungs=args.kalshi_reuse_rungs,
        )

    if args.backtest_fold_test:
        pipeline.walkForwardOverThresholds(
                 startDate = args.start_date,
                 endDate = args.end_date 
        )

    if args.pull_props:
        from betting.oddsCollector import pullHistoricalProps
        pullHistoricalProps(args.pull_props[0], args.pull_props[1], dbPath=args.db)

    if args.verify_totals:
        from betting.totalsCollector import verifyPull
        verifyDate = None if args.verify_totals == "__auto__" else args.verify_totals
        verifyPull(dbPath=args.db, date=verifyDate)

    if args.pull_totals:
        from betting.totalsCollector import pullHistoricalTotals
        pullHistoricalTotals(args.pull_totals[0], args.pull_totals[1],
                             dbPath=args.db, dryRun=args.totals_dry_run,
                             maxCredits=args.totals_max_credits)

    # Live / CLV validation workflow
    def _resolveDate(val):
        if val == "__today__":
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d")
        return val

    if args.freshness is not None:
        from betting.freshness import checkFreshness
        checkFreshness(args.db, _resolveDate(args.freshness))

    if args.snapshot_open:
        from betting.oddsCollector import snapshotLiveProps
        snapshotLiveProps(dbPath=args.db, snapshotType="open",
                          dryRun=args.snapshot_dry_run)

    if args.snapshot_close:
        from betting.oddsCollector import snapshotLiveProps
        snapshotLiveProps(dbPath=args.db, snapshotType="close",
                          dryRun=args.snapshot_dry_run)

    if args.score_live is not None:
        from betting.liveScorer import scoreLiveDay
        scoreLiveDay(dbPath=args.db, date=_resolveDate(args.score_live),
                     overThresh=args.over_edge_thresh,
                     underThresh=args.under_edge_thresh)

    if args.compute_clv is not None:
        from betting.liveScorer import computeCLV
        computeCLV(dbPath=args.db, date=_resolveDate(args.compute_clv))

    if args.clv_report is not None:
        from betting.liveScorer import clvReport
        start = args.clv_report[0] if len(args.clv_report) >= 1 else None
        end = args.clv_report[1] if len(args.clv_report) >= 2 else None
        clvReport(dbPath=args.db, startDate=start, endDate=end)


    if (args.kalshi_snapshot or args.kalshi_settle or args.kalshi_backfill is not None
            or args.kalshi_paper_decide or args.kalshi_paper_settle):
        from data.dbManager import DBManager
        DBManager(args.db).initSchema()

    if args.kalshi_snapshot:
        from betting.kalshiCollector import snapshotKalshiProps
        snapshotKalshiProps(dbPath=args.db, snapshotType=args.kalshi_snapshot,
                            windowMin=args.kalshi_window, dryRun=args.kalshi_dry_run)

    if args.kalshi_settle:
        from betting.kalshiCollector import settleKalshiProps
        throughDate = None if args.kalshi_settle == "__auto__" else args.kalshi_settle
        settleKalshiProps(dbPath=args.db, throughDate=throughDate, dryRun=args.kalshi_dry_run)

    if args.split_kalshi_db:
        from data.dbManager import splitKalshiDb
        splitKalshiDb(args.db)

    if args.kalshi_fit_shrink:
        from betting.kalshiPaper import fitShrinkWeight
        csvPath = None if args.kalshi_fit_shrink == "__paper__" else args.kalshi_fit_shrink
        fitShrinkWeight(dbPath=args.db, csvPath=csvPath, sinceDate=args.kalshi_shrink_since)

    if args.kalshi_paper_settle:
        from betting.kalshiPaper import settlePaperBets
        settlePaperBets(dbPath=args.db, dryRun=args.kalshi_dry_run)

    if args.kalshi_paper_decide:
        from betting.kalshiPaper import decidePaperBets
        decidePaperBets(dbPath=args.db, minLeadHours=args.kalshi_paper_lead[0],
                        maxLeadHours=args.kalshi_paper_lead[1], evThresh=args.kalshi_ev_thresh,
                        minPrice=args.kalshi_price_band[0], maxPrice=args.kalshi_price_band[1],
                        dryRun=args.kalshi_dry_run)

    if args.kalshi_paper_report is not None:
        from betting.kalshiPaper import paperReport
        start = args.kalshi_paper_report[0] if len(args.kalshi_paper_report) >= 1 else None
        end = args.kalshi_paper_report[1] if len(args.kalshi_paper_report) >= 2 else None
        paperReport(dbPath=args.db, startDate=start, endDate=end)

    if args.kalshi_backfill is not None:
        from betting.kalshiCollector import backfillKalshiArchive
        start = args.kalshi_backfill[0] if len(args.kalshi_backfill) >= 1 else None
        end = args.kalshi_backfill[1] if len(args.kalshi_backfill) >= 2 else None
        backfillKalshiArchive(dbPath=args.db, series=args.kalshi_series, startDate=start, endDate=end,
                              dryRun=args.kalshi_dry_run)

    if args.cache_data:
        pipeline.cacheFeatures()

    if args.refit_calibrator:
        pipeline.refitCalibrator()

    if args.evaluate:
        pipeline.evaluateModel()

    if args.calibrator:
        pipeline.evaluateCalibrator()
    
    if args.scrape:
        from data.scrapperEngine import ScrapeEngine
        from data.dbManager import DBManager

        db = DBManager(args.db)
        db.initSchema()
        engine = ScrapeEngine(db=args.db, headless=True)

        try:
            _runScrape(engine, db, args)
        finally:
            engine.close()

    if args.historical_seasons:
        scrapeHistorical(args.historical_seasons, dbPath=args.db, outputDir="output")

    if args.scrape_ratings is not None:
        scrapeRatings(seasons=args.scrape_ratings, dbPath=args.db)

    if args.train_minutes:
        import sqlite3
        from features.cache import preloadCaches
        from models.minutes import MinutesBundle

        conn = sqlite3.connect(args.db)
        caches = preloadCaches(conn)
        conn.close()
        MinutesBundle.train(
            playerLogCache = caches.playerLogCache,
            statusDF = caches.statusDF,
            posCache = caches.posCache,
            dbPath = args.db,
            endDate = args.train_end_date,
            save = True
        )

    if args.cache_results_features:
        pipeline.cacheResultsFeatures()

    if args.train_results:
        pipeline.trainResults(endDate=args.train_end_date)

    if args.eval_results_calibrator:
        pipeline.evaluateResultsCalibrator()

    if args.eval_results_layers:
        pipeline.evaluateResultsLayers(endDate=args.train_end_date,
                                       lineHavingSplit=args.line_having_split)


def _runScrape(engine, db, args):
    import json
    outputDir = "output"
    numLogGames = args.num_games
    backfillFrom = args.backfill_from
    try:
        print("\n--------Scraping--------")
        teams = engine.scrapeTeams()

        players = engine.scrapePlayersHistorical(seasons=[CURRENT_SEASON])
        with open(f"{outputDir}/players.json", "w") as f:
            json.dump(players, f, indent=2)
        
        games = engine.scrapeGames()
        with open(f"{outputDir}/games.json", "w") as f:
            json.dump(games, f, indent=2)

        results = engine.scrapeResults()

        teamGameRatings = engine.scrapeTeamGameRatings()

        logs = engine.scrapeLogs(numGames=numLogGames)
        
        if backfillFrom:
            status = engine.scrapeStatusRange(backfillFrom)
        else:
            status = engine.scrapeStatusAutoFill()
        
    finally:
        engine.close()

    print("\n--------Loading Items into DB--------")

    db.upsertTeams(teams)
    db.upsertPlayers(players)
    db.upsertGames(games)
    db.upsertResults(results)
    db.upsertTeamGameRatings(teamGameRatings)
    db.upsertLogs(logs)
    db.upsertStatus(status)

    print("\n--------DB Updated Complete--------")

def scrapeHistorical(seasons, dbPath="NBA.db", outputDir="output"):
    import json
    from data.scrapperEngine import ScrapeEngine
    from data.dbManager import DBManager
    # Setup DB and Scrapper Engine
    db = DBManager(dbPath)
    db.initSchema()
    engine = ScrapeEngine(db=dbPath, headless=True)

    try:
        print(f"\n-------- Historical Scrape: {seasons} --------")
        
        # Teams
        teams = engine.scrapeTeamsHistorical(seasons)
        db.upsertTeams(teams)

        # Players
        players = engine.scrapePlayersHistorical(seasons)
        with open(f"{outputDir}/players.json", "w")  as f:
            json.dump(players, f, indent=2)
        db.upsertPlayers(players)

        # Games and logs
        allGames = []
        allResults = []
        for season in seasons:
            games = engine.scrapeGames(season=season)
            allGames.extend(games)
            results = engine.scrapeResults(season=season)
            allResults.extend(results)
        with open(f"{outputDir}/games.json", "w") as f:
            json.dump(allGames, f, indent=2)
        db.upsertGames(allGames)
        db.upsertResults(allResults)

        # Per-game ratings need games.json written first (scrapeTeamGameRatings reads it)
        for season in seasons:
            ratings = engine.scrapeTeamGameRatings(season=season)
            db.upsertTeamGameRatings(ratings)

        for season in seasons:
            logs = engine.scrapeLogsHistorical(season=season)
            db.upsertLogs(logs)

        # Status
        for season in seasons:
            startDate = f"{season - 1}-10-01"
            endDate = f"{season}-06-30"
            status = engine.scrapeStatusRange(startDate, endDate)
            db.upsertStatus(status)

    finally:
        engine.close()

def scrapeRatings(seasons=None, dbPath="NBA.db"):
    """
    Backfills per-game team ratings (off/def rtg, pace) from the NBA stats API.

    seasons: list of ints, or None/empty for all seasons in games.json.
    Reads games.json, so run a games scrape first if it's stale.
    The NBA stats API blocks many datacenter IPs, so run this from a machine
    that can reach stats.nba.com.
    """
    from data.scrapperEngine import ScrapeEngine
    from data.dbManager import DBManager

    db = DBManager(dbPath)
    db.initSchema()

    # scrapeTeamGameRatings only needs _loadJson + requests, so skip the Selenium
    # driver setup (and the teams/players lookups it builds) entirely.
    engine = ScrapeEngine.__new__(ScrapeEngine)

    seasonList = seasons if seasons else [None]  # None -> all seasons
    total = 0
    for season in seasonList:
        label = "all seasons" if season is None else f"season {season}"
        print(f"\n-------- Scraping ratings: {label} --------")
        ratings = engine.scrapeTeamGameRatings(season=season)
        db.upsertTeamGameRatings(ratings)
        total += len(ratings)

    print(f"\n-------- Ratings scrape complete: {total} rows --------")

if __name__ == "__main__":
    main()

# :steam_smile
