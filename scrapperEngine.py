import time
import re
import json
import unicodedata
import sqlite3
from datetime import datetime
from bs4 import BeautifulSoup
from seleniun import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Add docstings fat bum

class scrapeEngine:
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
        if (headless):
            options = options.add_argument('--headless')

        options = options.add_argument('--disable-blink-features=AutomationControlled')
        service = Service(ChromeDriverManager().install())

        return webdriver.Chrome(service=service, options=options)

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
    def _init_playerLookup(self, name):
        # Load the players just file as a list
        players = self._loadJson(self, "output/players.json")

        # Loop through list and build a dictonary where the key is the normalized name the value is the player id
        lookup = {}

        for p in players:
            normalized = self._normalizeName(p["name"])
            lookup[normlized] = p["player_id"]

        return lookup

    #
    def _convertMins(self, minString):
        if not minString or ':' not in minString:
            return 0.0

        # Maps mins and strings on the string param by splitting on the :, then rounds and returns as one value
        m, s = map(int, minString.split(':'))
        return round(m + (s / 60.0), 2)
        


    # SCRAPPERS

    def scrapeGames(self, season=2026):
        """
        Overveiw:
            Scrapes basketball reference for all games in specifed season.
        Params
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
            for url in monthLinks:
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

                # FIXME: Understand.....
                rows = table.find("tbody").find_all(
                        "tr", class_=lambda x: x != "thread"
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
                    awayAbbr = self.fullNameConversion.get(awayTD.text.strip())

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
        url = "https://www.basketball-reference.com/boxscores"
        logs = []
        restLookup = {}
        lastLogDate = {}

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

        





        







                        


                
            

    
    # FIXME: Add docstring
    def scrapeLogs():
        pass

    # FIXME: Add docstring
    def scrapePlayers():
        pass
    
    # FIXME: Add docstring
    def scrapeResults():
        pass
    
    # FIXME: Add docstring
    def scrapeStatus():
        pass
    
    # FIXME: Add docstring
    def scrapeTeams():
        pass

    
    # Gets the last date data was added to db
    def getLastScrapeDate(self):
        conn = sqlite3.connect(self.db)
        # FIXME: Might have tp change later
        res = conn.execute("SELECT MAX(game_date) FROM Games WHERE home_score IS NOT NULL").fetchone()
        conn.close()

        if res[0]:
            return res[0]
        else:
            "2025-10-01"

    # Shuts down the driver
    def close(self):
        self.driver.quit()


