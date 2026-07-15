import argparse
from pipeline.orchestrator import Pipeline

def main():
    parser = argparse.ArgumentParser(description="NBA prediction proj")
 
    # Scrape args
    parser.add_argument("--scrape", action="store_true")
    parser.add_argument("--backfill-from", type=str, default=None)
    parser.add_argument("--num-games",  type=int, default=None)
    parser.add_argument("--historical-seasons", type=int, nargs="+", default=None)
    
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

    # Props args
    parser.add_argument("--pull-props", nargs=2, metavar=("START_DATE", "END_DATE"))

    # Live / CLV validation args
    parser.add_argument("--freshness", nargs="?", const="__today__", metavar="DATE")
    parser.add_argument("--snapshot-open", action="store_true")
    parser.add_argument("--snapshot-close", action="store_true")
    parser.add_argument("--snapshot-dry-run", action="store_true")
    parser.add_argument("--score-live", nargs="?", const="__today__", metavar="DATE")
    parser.add_argument("--compute-clv", nargs="?", const="__today__", metavar="DATE")
    parser.add_argument("--clv-report", nargs="*", metavar=("START", "END"))

    # Backtest args
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--backtest-unders", action="store_true")
    parser.add_argument("--backtest-combined", action="store_true")
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

    if args.backtest_fold_test:
        pipeline.walkForwardOverThresholds(
                 startDate = args.start_date,
                 endDate = args.end_date 
        )

    if args.pull_props:
        from betting.oddsCollector import pullHistoricalProps
        pullHistoricalProps(args.pull_props[0], args.pull_props[1], dbPath=args.db)

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


def _runScrape(engine, db, args):
    import json
    outputDir = "output"
    numLogGames = args.num_games
    backfillFrom = args.backfill_from
    try:
        print("\n--------Scraping--------")
        teams = engine.scrapeTeams()

        players = engine.scrapePlayersHistorical(seasons=[2026])
        with open(f"{outputDir}/players.json", "w") as f:
            json.dump(players, f, indent=2)
        
        games = engine.scrapeGames()
        with open(f"{outputDir}/games.json", "w") as f:
            json.dump(games, f, indent=2)

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
        for season in seasons:
            games = engine.scrapeGames(season=season)
            allGames.extend(games)
        with open(f"{outputDir}/games.json", "w") as f:
            json.dump(allGames, f, indent=2)
        db.upsertGames(allGames)

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

if __name__ == "__main__":
    main()

# :steam_smile
