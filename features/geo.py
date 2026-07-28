"""
geo.py

Static geography for the 30 NBA teams' home arenas, used by the totals model's
travel / altitude features. The Games table has no venue or coordinate data, but
a team's home arena is fixed, so home_team_id -> venue is a reliable lookup.

VENUES maps team_id (see Teams table) -> (lat, lon, tz_offset, altitude_ft):
  lat/lon     decimal degrees of the arena, for great-circle distance
  tz_offset   standard-time UTC offset (hours). We don't model DST — the diff
              between two venues is what matters and DST cancels out for all but
              Arizona, whose ~0.5-game-a-year effect isn't worth the complexity.
  altitude_ft arena elevation. Only Denver and Utah are meaningfully high; the
              rest are near sea level and only matter as the "from" end of a trip.

Sources: arena coordinates and city elevations (rounded to the arena, not the
downtown marker). Values are static — arenas don't move between seasons.
"""

from math import radians, sin, cos, asin, sqrt

# team_id -> (lat, lon, tz_offset_hours, altitude_ft)
VENUES = {
    1:  (39.7487, -105.0077, -7, 5280),  # DEN - Ball Arena (high altitude)
    2:  (35.4634,  -97.5151, -6,  1201),  # OKC - Paycom Center
    3:  (29.7508,  -95.3621, -6,    50),  # HOU - Toyota Center
    4:  (40.7505,  -73.9934, -5,    33),  # NYK - Madison Square Garden
    5:  (25.7814,  -80.1870, -5,     7),  # MIA - Kaseya Center
    6:  (29.4270,  -98.4375, -6,   650),  # SAS - Frost Bank Center
    7:  (40.7683, -111.9011, -7,  4265),  # UTA - Delta Center (high altitude)
    8:  (44.9795,  -93.2760, -6,   830),  # MIN - Target Center
    9:  (34.0430, -118.2673, -8,   285),  # LAL - Crypto.com Arena
    10: (42.6966,  -83.2454, -5,   660),  # DET - Little Caesars Arena
    11: (45.5316, -122.6668, -8,    50),  # POR - Moda Center
    12: (41.4965,  -81.6882, -5,   650),  # CLE - Rocket Mortgage FieldHouse
    13: (41.8807,  -87.6742, -6,   594),  # CHI - United Center
    14: (28.5392,  -81.3839, -5,    82),  # ORL - Kia Center
    15: (33.7573,  -84.3963, -5,  1050),  # ATL - State Farm Arena
    16: (39.9012,  -75.1720, -5,    39),  # PHI - Wells Fargo Center
    17: (42.3662,  -71.0621, -5,    19),  # BOS - TD Garden
    18: (35.2251,  -80.8392, -5,   751),  # CHO - Spectrum Center
    19: (43.6435,  -79.3791, -5,   250),  # TOR - Scotiabank Arena
    20: (29.9490,  -90.0821, -6,     3),  # NOP - Smoothie King Center
    21: (35.1382,  -90.0505, -6,   337),  # MEM - FedExForum
    22: (33.4457, -112.0712, -7,  1086),  # PHO - Footprint Center
    23: (37.7680, -122.3877, -8,    13),  # GSW - Chase Center
    24: (43.0451,  -87.9172, -6,   617),  # MIL - Fiserv Forum
    25: (32.7905,  -96.8103, -6,   430),  # DAL - American Airlines Center
    26: (38.8981,  -77.0209, -5,    25),  # WAS - Capital One Arena
    27: (38.5802, -121.4997, -8,    30),  # SAC - Golden 1 Center
    28: (34.0430, -118.2673, -8,   285),  # LAC - Intuit Dome / shares LA
    29: (39.7640,  -86.1555, -5,   715),  # IND - Gainbridge Fieldhouse
    30: (40.6826,  -73.9754, -5,    33),  # BRK - Barclays Center
}

# Arenas above this are treated as thin-air games (Denver, Utah).
ALTITUDE_THRESHOLD_FT = 3500


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in miles."""
    r = 3958.8  # earth radius, miles
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def travelMiles(fromTeamID, toTeamID):
    """Miles between two teams' home venues. 0 if either is unknown or they match."""
    a = VENUES.get(int(fromTeamID))
    b = VENUES.get(int(toTeamID))
    if a is None or b is None:
        return 0.0
    return haversine(a[0], a[1], b[0], b[1])


def tzChange(fromTeamID, toTeamID):
    """Signed timezone-offset difference (to - from) in hours between two venues.
    Positive = travelling east (losing hours, e.g. west coast -> east coast)."""
    a = VENUES.get(int(fromTeamID))
    b = VENUES.get(int(toTeamID))
    if a is None or b is None:
        return 0.0
    return float(b[2] - a[2])


def isHighAltitude(teamID):
    """True if this team hosts at high altitude (Denver / Utah)."""
    v = VENUES.get(int(teamID))
    return bool(v and v[3] >= ALTITUDE_THRESHOLD_FT)
