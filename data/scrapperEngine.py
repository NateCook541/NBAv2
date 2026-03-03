import time
import re
import json
import unicodedata
import sqlite3
from datetime import date
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

# Add docstings fat bum

class ScrapeEngine:
    # db path defaults to NBA.db
    # headless to make google not pop up during scrapping
    def __init__(self, db='NBA.db', headless=True):
        self.db = db
        self.driver = self._setupDriver(headless)

        self.teamMap = self._loadJson("output/teams_map.json")
        self.fullNameConversion = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
            "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
        }

        self.playerLookup = self._init_playerLookup()

    # PRIVATE METHODS

    # Sets up the driver to be used in scrapping functions, gives basic options along with a true/false headless option
    def _setupDriver(self, headless):
        options = Options()

        if headless:
            options.add_argument("--headless=new")  # modern headless mode

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        driver = webdriver.Chrome(options=options)

        driver.set_page_load_timeout(30)
        return driver

    # Trys to open and read json file, if fails return empty list    
    def _loadJson(self, path): 
        try:
            with open(path, 'r', encoding='utf-8') as file:
                    return json.load(file)
        except:
            return {}

    # Returns a name normalized to remove accents and stuff (Exg luka)
    def _normalizeName(self, name):
        return "".join(c for c in unicodedata.normalize('NFD', name) 
                        if unicodedata.category(c) != 'Mn').lower().strip() 
    
    # Finds and players name in the db and returns it normalized
    def _init_playerLookup(self):
        # Load the players just file as a list
        players = self._loadJson("output/players.json")

        # Loop through list and build a dictonary where the key is the normalized name the value is the player id
        lookup = {}

        for p in players:
            normalized = self._normalizeName(p["name"])
            lookup[normalized] = p["player_id"]

        return lookup

    #
    def _convertMins(self, minString):
        if not minString or ':' not in minString:
            return 0.0

        # Maps mins and strings on the string param by splitting on the :, then rounds and returns as one value
        m, s = map(int, minString.split(':'))
        return round(m + (s / 60.0), 2)
        
    # Strips comments in html page using re
    def _stripComments(self, html):
        return re.sub(r"<!--(.*?)-->", r"\1", html, flags=re.DOTALL)

    # SCRAPPERS

    def scrapeGames(self, season=2026):
        """
        Overveiw:
            Scrapes basketball reference for all games in specifed season.
        Params:
            season: int of season to scrape
        Returns:
            A list containing game_id, game_date, home_team_id, away_team_id, and season for each game in a season.
        """

        # Get the url and use a f string as we need to define which season to scrape
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
        allGames = []

        try:
            print(f"Opening URL - {url}")
            self.driver.get(url)

            # FIXME: understand....
            WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "filter"))
            )

            # Start up beautifulSoup to parse the HTML
            soupInit = BeautifulSoup(self.driver.page_source, "html.parser")
        
            # Filter the div elements in the html source 
            filterDiv = soupInit.find("div", class_="filter")

            # FIXME: understand....
            if filterDiv:
                monthLinks = filterDiv.find_all(
                    "a", href=re.compile(r"games-.*\.html")
                )
                monthURLs = [url + link["href"] for link in monthLinks]
            else:
                monthURLs = [url]

            # Go over each month found in the main url HTML source
            for url in monthURLs:
                monthLabel = url.split('-')[-1].replace('.html', "").capitalize()
                print(f"---------Scraping {monthLabel}----------")

                self.driver.get(url)
                time.sleep(2)

                # Clean the HTML by removing comments using re package and five the clean HTML to beatufiulSoup to parse
                cleanHTML = re.sub(r"<!--|-->", "", self.driver.page_source)
                soup = BeautifulSoup(cleanHTML, "html.parser")
                
                # Find the schedule table and if no table for that month found skip over and continue
                table = soup.find("table", {"id": "schedule"})
                if not table:
                    print(f"No table found for {monthLabel}")
                    continue

                # Gets all rows in the page to loop and pull info for
                rows = table.find("tbody").find_all(
                        "tr", class_= lambda x: x != "thread"
                )

                # Go over all rows in that months table
                for row in rows:
                    # Gets the game date and skips if not found
                    dateTH = row.find("th", {"data-stat": "date_game"})
                    gameID = dateTH.get("csk") if dateTH else None
                    if not gameID:
                        continue

                    # Gets the visitor and home team name and skips if not found
                    visitorTD = row.find("th", {"data-stat": "visitor_team_name"})
                    homeTD = row.find("th", {"data-stat": "home_team_name"})
                    if not visitorTD or not homeTD:
                        continue
                    
                    # Converts the name to the abbrvation used in the main team map to get the correct ID. Just continues if empty
                    homeAbbr = self.fullNameConversion.get(homeTD.text.strip())
                    awayAbbr = self.fullNameConversion.get(visitorTD.text.strip())

                    homeID = self.teamMap.get(homeAbbr)
                    awayID = self.teamMap.get(awayAbbr)

                    if homeID is None or awayID is None:
                        continue
                    
                    # Gets the date and converts y/m/d format
                    # Then in the same try block builds the game list
                    dateText = dateTH.text.strip()
                    try:
                        dateParaOne = datetime.strptime(
                                dateText, "%a, %b %d, %y"
                        )

                        allGames.append({
                            "game_id": game_id,
                            "game_date": dateParaOne.strftime("%Y-%m-%d"),
                            "home_team_id": int(homeID),
                            "away_team_id": int(awayID),
                            "season": season
                        })

                    except ValueError:
                        continue

                print(f"Games scrapped so far: {len(allGames)}")
                
            print(f"Total games scrapped {len(allGames)}")
            return allGames

        except Exception as e:
            print(f"Error during games scrapping {e}")
            return []

    
    def scrapeLogs(self, numGames=None):
        """
        Overveiw:
            Scrapes basketball reference for all player logs for each game in a season.
        Params
            numGames: Optional parameter to decide how many games you wanrt to scrape. Its default is None, so lasScrapeDate is called to find out how many games to scrape instead 
        Returns:
            A list containing logs for each player for each game, holding important stats such as points, rebounds, etc...
        """

        urlMain = "https://www.basketball-reference.com/boxscores"
        logs = []
        restLookup = {}
        lastGameDate = {}

        # Get the last date logs were added to the db and get todays date in y-m-d format
        lastScrapped = self.getLastScrapeDate()
        today = datetime.now().strftime("%Y-%m-%d")

        # Create a list of games to scrape from games.json by checking the current date and the last scraped date
        games = self._loadJson("output/games.json")
        gamesList = [
                g for g in games
                if lastScrapped < g["game_date"] <= today
        ]
        
        if not gamesList:
            print("Logs up to date")
            return []

        if numGames:
            gamesList = gamesList[:numGames]
        
        # Creates the sorted games list, which sorts the games needing logs scrapped by their date
        sortedGames = sorted(gamesList, key=lambda x: x["game_date"])
        for g in sortedGames:
            currentDate = datetime.strptime(g["game_date"], "%Y-%m-%d")
            # FIXME: Understand...
            for side in ("home_team_id", "away_team_id"):
                teamid = g[side]
                restLookup[(g["game_id"], teamid)] = (
                        (currentDate - lastGameDate[teamid]).days
                        if teamid in lastGameDate else 20
                )
                lastGameDate[teamid] = currentDate

        # Loops through each game in gameslist to get each log
        for game in gamesList:
            gameID = game["game_id"]
            homeTeamID = game["home_team_id"]
            url = f"{url}/{gameID}.html"
            print(f"Scrapping log {url}")

            try:
                # Get the url for the driver and sleeps to not make site agery
                self.driver.get(url)
                self.driver.sleep(5)

                # Gets the data from the tables on each game log without comments to loop through to get indivudal player stats
                soup = BeautifulSoup(self._stripComments(self.driver.page_source), "html.parser")
                tables = soup.find_all("table", id=re.compile(r"box-[A-Z]{3}-game-basic"))

                for table in tables:
                    # This chuck gets the team stats and maps it using the given team map in the class vars
                    abbrMatch = re.search(r"box-([A-Z]{3})-game-basic", table["id"])
                    if not abbrMatch:
                        continue
                    
                    abbr = abbrMatch.group(1)
                    currentTeamID = self.teamMap.get(abbr)
                    if not currentTeamID:
                        continue
                    currentTeamID = int(currentTeamID)

                    isHome = currentTeamID == homeTeamID
                    daysRest = restLookup.get((gameID, currentTeamID), 0)
                    onStarters = True

                    for row in table.find("tbody").find_all("tr"):
                        if "thead" in row.get("class", []):
                            onStarters = False
                            continue

                        if row.find("td", {"data-stat": "reason"}):
                            continue

                        nameTH = row.find("th", {"data-stat": "player"})
                    if not nameTH:
                        continue
                    
                    # Gets the player name to pull their stats
                    cleanName = self._normalizeName(nameTH.get_text())
                    playerID = self.playerLookup.get(cleanName)
                    if playerID is None:
                        print(f"Player: {cleanName} not in lookup")
                        continue
                    
                    # Returns each listed stats for each player in the table
                    def getStat(stat, default=0, asFloat=False):
                        cell = row.find("td", {"data-stat": stat})
                        val = cell.get_text().strip if cell else ""
                        if not val or val == ".":
                            return default
                        return float(val) if asFloat else int(val)
                    # For each player get the stat amnd append it to the logs list
                    mpCell = row.find("td", {"data-stat": "mp"})
                    logs.append({
                        "log_id": None,
                        "player_id": playerID,
                        "game_id": gameID,
                        "minutes": self._convertMins(mpCell.get_text() if mpCell else ""),
                        "points": getStat("pts"),
                        "rebounds": getStat("trb"),
                        "assists": getStat("ast"),
                        "steals": getStat("stl"),
                        "blocks": getStat("blk"),
                        "turnovers": getStat("tov"),
                        "fg_pct": getStat("fg_pct", default=0.0, asFloat=True),
                        "is_starter": onStarters,
                        "is_home": isHome,
                        "rest_days": daysRest,
                    })

            except Exception as e:
                print(f"Failed to scrape game {gameID}: {e}")
                time.sleep(10)
        
        # Return final list of all logs
        print(f"Total logs scraped: {len(logs)}")
        return logs
        
    def scrapePlayers(self):
        """
        Overveiw:
            Scrapes NBA.com for a list of all active players along with their team and position.
        Params
            None
        Returns:
            A list containing active players and there information
        """

        url = "https://www.nba.com/players"
        players = []
        seenHrefs = set()
        nextID = 1
    
        # Gets the href anchors for each element (player on the site to process)
        def processAnchor(anchor):
            nonlocal nextID
            try:
                href = anchor.get_attribuite("href") or ""
            except Exception:
                return

            if not href or href in seenHrefs:
                return

            # Gets the name of players by searching through anchors
            name = ""
            try:
                container = anchor.find_element(By.CSS_SELECTOR,"div.RosterRow_playerName__G28lg")
                parts = container.find_elements(By.TAG_NAME, "p")
                name  = " ".join(p.text.strip() for p in parts if p.text.strip())
            except Exception:
                name = anchor.text.strip()
            if not name:
                return

            # Rows hold team + position data I need
            row = None
            for xpath in ("./ancestor::tr", "./ancestor::div[contains(@class,'RosterRow')]"):
                try:
                    row = anchor.find_element(By.XPATH, xpath)
                    break
                except Exception:
                    continue

            # Get the team abbrvations
            teamAbbr = None
            try:
                source   = row if row else anchor
                teamAbbr = source.find_element(By.CSS_SELECTOR, "a[href*='/team/']").text.strip()
            except Exception:
                pass

            # Get player positions
            position = None
            try:
                if row:
                    tds = row.find_elements(By.TAG_NAME, "td")
                    if len(tds) >= 4:
                        candidate = tds[3].text.strip()
                        if candiate and re.match(r"^[A-Za-z\-]{1,4}$", candidate):
                            position = candidate
            except Exception:
                pass

            # Get the team id by using the abbrvation and the team map class var
            teamID = int(self.teamMap[teamAbbr]) if teamAbbr and teamAbbr in self.teamMap else None
            
            # Append stats to the players list and mark the href as seen
            players.append({
                "player_id": nextID,
                "name": name,
                "team_id": teamID,
                "position": position,
                "is_active": True,
            })
            seenHrefs.add(href)

            print(f"  [{nextID}] {name} -- team:{teamAbbr} pos:{position}")
            nextID += 1

        # FIXME: Understand...
        try:
            print(f"Opening URL -- {url}")
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href*='/player/']"))
            )

            try:
                selElem = self.driver.find_element(
                        By.CSS_SELECTOR,
                        "select.DropDown_select__4pIg9, select[title*='Page Number']",
                )
                sel = Select(selElem)
                try:
                    sel.select_by_value("-1")
                    time.sleep(2)
                    WebDriverWait(self.driver, 10).until(
                        lambda d: len(d.find_elements(By.CSS_SELECTOR, "a[href*='/player/']")) > 50
                    )
                    for a in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/player/']"):
                        processAnchor(a)
                except Exception:
                    for opt in sel.options:
                        val = opt.get_attribute("value")
                        if val == "-1":
                            continue
                        try:
                            sel.select_by_value(val)
                            time.sleep(1)
                            WebDriverWait(self.driver, 8).until(
                                lambda d: len(d.find_elements(By.CSS_SELECTOR, "a[href*='/player/']")) > 0
                            )
                            for a in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/player/']"):
                                processAnchor(a)
                        except Exception:
                            continue
            except Exception:
                for a in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/player/']"):
                    processAnchor(a)

        except Exception as e:
            print(f"Error in scrapePlayers: {e}")
        
        self.playerLookup = {self._normalizeName(p["name"]): p["player_id"] for p in players}

        print(f"Total players scraped: {len(players)}")
        return players

    def scrapeTeams(self):
        """
        Overveiw:
            Scrapes basketball reference for a list of all teams and there offensive and defensive ratings.
        Params
            None
        Returns:
            A list containing teams with there offensive and defensive stats
        """

        url = "https://www.basketball-reference.com/leagues/NBA_2026.html"
        teamsOut = []
        today = date.today().isoformat()
        seen = set()

        try:
            print(f"Opening URL -- {url}")
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "td[data-stat='team] a"))
            )
            
            # Strip comments as the advanced talbe is hidden inside them
            soup = BeautifulSoup(self._stripComments(self.driver.page_source), "html.parser")
            table = soup.find("table", {"id": "advanced-team"})
            if not table:
                print("Advanced-team table not found")
                return []

            for row in table.find("tbody").find_all("tr"):
                if row.get("class") and "thread" in row["class"]:
                    continue

                teamTD = row.find("td", {"data-stat": "team"})
                if not teamID:
                    continue

                a = teamTD.find("a")
                if not a:
                    continue

                href = a.get("href", "")
                abbrMatch = re.search(r"/teams/([A-Z]{2,4})/", href)
                abbr = abbrMatch.group(1) if abbrMatch else None
                if not abbr or abbr in seen:
                    continue

                # FIXME: Understand...
                def cell(stat):
                    td = row.find("td", {"data-stat": stat})
                    if td and td.get_text().strip():
                        try:
                            return float(td.get_text().strip())
                        except ValueError:
                            pass
                    return None
                
                # Appends team stats and info to teamsOut list
                teamID = self.teamMap.get(abbr)
                teamsOut.append({
                    "team_id": int(teamID) if teamID is not None else None,
                    "name": abbr,
                    "off_rtg": cell("off_rtg"),
                    "def_rtg": cell("def_rtg"),
                    "pace": cell("pace"),
                    "date": today,
                })
                seen.add(abbr)
                print(f"  {abbr} -> off:{teamsOut[-1]['off_rtg']} def:{teamsOut[-1]['def_rtg']} pace:{teamsOut[-1]['pace']}")

        except Exception as e:
            print(f"Error in scrapeTeams: {e}")

        
        print(f"Total teams scraped: {len(teamsOut)}")
        return teamsOut


    def scrapeStatus(self):
        """
        Overveiw:
            Scrapes epsn for a list of all current injured players and lists their estimated return date, along with details of the injury. Also adds the scrape date as the data needs to be time series. 
        Params
            None
        Returns:
            A list containing injured players along with other information.
    """
        url = "https://www.espn.com/nba/injuries"
        statusData = []
        nextLogID = 1
        today = datetime.now().strftime("%Y-%m-%d")

        games = self._loadJson("output/games.json")
        if isinstance(games, dict):
            games = []
        nextGameLookup = {}
        for game in sorted(games, key=lambda g: g["game_date"]):
            if game["game_date"] >= today:
                for side in ("home_team_id", "away_team_id"):
                    tid = game[side]
                    if tid not in nextGameLookup:
                        nextGameLookup[tid] = game["game_id"]

        # Oepn the url and start to parse with soup
        try:
            print(f"Opening injuries url -- {url}")
            self.driver.get(url)
            time.sleep(5)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            
            # FIXME: Understand...
            for section in soup.find_all("div", class_="ResponsiveTable"):
                titleDiv = section.find("div", class_="Table__Title")
                espnName = titleDiv.get_text().strip() if titleDiv else None
                abbr = self.fullNameConversion.get(espnName)
                rawID = self.teamMpa.get(abbr) if abbr else None
                if not rawID:
                    continue
                teamID = int(raw)
                targetGameID = nextGameLookup.get(teamID)

                tbody = section.find("tbody", class_="Table__TBODY")
                if not tbody:
                    continue

                # FIXME: Understand...
                for row in tbody.find_all("tr", class_="Table__TR"):
                    nameTD = row.find("td", class_="col-name")
                    if not nameTD:
                        continue

                    cleanName = self._normalizeName(nameTD.get_text().strip())
                    playerID = self.playerLookup.get(cleanName)
                    if playerID is None:
                        print(f"Player {cleanName} ({abbr}) not in lookup")
                        continue

                    def col(cls):
                        c = row.find("td", class_=cls)
                        return c.get_text().strip() if c else ""
                    
                    statusData.append({
                        "status_log_id": nextLogID,
                        "player_id": playerID,
                        "team_id": teamID,
                        "game_id": targetGameID,
                        "scrape_date": today,
                        "status": col("col-stat"),
                        "return_date": col("col-date"),
                        "comment": col("col-desc"),
                    })
                    nextLogID += 1

                print(f"Total status records scraped: {len(statusData)}")

        except Exception as e:
            print(f"Error in scrapeStatus: {e}")

        return statusData

    def scrapeResults(self):
        """
        Overveiw:
            Scrapes basketball reference and scrapes which team won eacg game in a season, along with the total scores for each team
        Params
            None
        Returns:
            A list containing results of the games
        """
        url = "https://www.basketball-reference.com/leagues/NBA_2026_games.html"
        results = []

        try:
            print(f"Opening {startURL}")
            self.driver.get(startURL)
            WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "filter"))
            )

            soupInit = BeautifulSoup(self.driver.page_source, "html.parser")
            filterDiv = soupInit.find("div", class_="filter")
            monthLinks = filterDiv.find_all("a", href=re.compile(r"games-.*\.html"))
            monthURLs = [url + link["href"] for link in monthLinks] or [url]

            for monthURL in monthURLs:
                self.driver.get(monthURL)
                time.sleep(2)
                soup = BeautifulSoup(self._stripComments(self.driver.page_source), "html.parser")
                table = soup.find("table", {"id": "schedule"})
                if not table:
                    continue

                for row in table.find("tbody").find_all("tr"):
                    if row.get("class") and "thead" in row["class"]:
                        continue

                    visitorPtsTD = row.find("td", {"data-stat": "visitor_pts"})
                    homePtsTD    = row.find("td", {"data-stat": "home_pts"})
                    if not visitorPtsTD or not visitorPtsTD.text.strip():
                        continue

                    try:
                        visitorPts = int(visitorPtsTD.text)
                        homePts    = int(homePtsTD.text)
                    except (ValueError, AttributeError):
                        continue

                    dateTH = row.find("th", {"data-stat": "date_game"})
                    gameID = dateTH.get("csk") if dateTH else None
                    if not gameID:
                        continue

                    homeName = (row.find("td", {"data-stat": "home_team_name"}) or {}).text.strip()
                    awayName = (row.find("td", {"data-stat": "visitor_team_name"}) or {}).text.strip()
                    homeAbbr = self.fullNameConversion.get(homeName)
                    awayAbbr = self.fullNameConversion.get(awayName)
                    homeID   = int(self.teamMap[homeAbbr]) if homeAbbr and homeAbbr in self.teamMap else None
                    awayID   = int(self.teamMap[awayAbbr]) if awayAbbr and awayAbbr in self.teamMap else None
                    if not homeID or not awayID:
                        continue

                    results.append({
                        "game_id":      gameID,
                        "home_team_id": homeID,
                        "away_team_id": awayID,
                        "home_score":   homePts,
                        "away_score":   visitorPts,
                        "winner_id":    homeID if homePts > visitorPts else awayID,
                    })

            print(f"Total results scraped: {len(results)}")

        except Exception as e:
            print(f"Error in scrapeResults: {e}")

        return results

    def close(self):
        """
            Overview:
                Shuts down the web driver
            Params:
                None
            Return:
                None
        """
        self.driver.quit()
   
    # Gets the last date data was added to db
    def getLastScrapeDate(self):
        conn = sqlite3.connect(self.db)
        # FIXME: Might have tp change later
        res = conn.execute("SELECT MAX(game_date) FROM Games WHERE game_date IS NOT NULL").fetchone()
        conn.close()

        if res[0]:
            return res[0]
        else:
            "2025-10-01"

