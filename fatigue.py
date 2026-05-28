from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

try:
    from .core.utils import clamp, names_match
except ImportError:
    from core.utils import clamp, names_match


NBA_TEAM_COORDS = {
    "Atlanta Hawks": (33.7573, -84.3963),
    "Boston Celtics": (42.3662, -71.0621),
    "Brooklyn Nets": (40.6826, -73.9754),
    "Charlotte Hornets": (35.2251, -80.8392),
    "Chicago Bulls": (41.8807, -87.6742),
    "Cleveland Cavaliers": (41.4965, -81.6882),
    "Dallas Mavericks": (32.7905, -96.8103),
    "Denver Nuggets": (39.7487, -105.0077),
    "Detroit Pistons": (42.3410, -83.0550),
    "Golden State Warriors": (37.7680, -122.3877),
    "Houston Rockets": (29.7508, -95.3621),
    "Indiana Pacers": (39.7640, -86.1555),
    "Los Angeles Clippers": (34.0430, -118.2673),
    "Los Angeles Lakers": (34.0430, -118.2673),
    "Memphis Grizzlies": (35.1382, -90.0505),
    "Miami Heat": (25.7814, -80.1870),
    "Milwaukee Bucks": (43.0451, -87.9172),
    "Minnesota Timberwolves": (44.9795, -93.2761),
    "New Orleans Pelicans": (29.9490, -90.0821),
    "New York Knicks": (40.7505, -73.9934),
    "Oklahoma City Thunder": (35.4634, -97.5151),
    "Orlando Magic": (28.5392, -81.3839),
    "Philadelphia 76ers": (39.9012, -75.1720),
    "Phoenix Suns": (33.4457, -112.0712),
    "Portland Trail Blazers": (45.5316, -122.6668),
    "Sacramento Kings": (38.5802, -121.4997),
    "San Antonio Spurs": (29.4269, -98.4375),
    "Toronto Raptors": (43.6435, -79.3791),
    "Utah Jazz": (40.7683, -111.9011),
    "Washington Wizards": (38.8981, -77.0209),
}


@dataclass(frozen=True)
class FatigueProfile:
    """NBA fatigue features for one team entering a target game."""

    team: str
    rest_days: int
    back_to_back: bool
    travel_penalty: float
    fatigue_score: float
    warning: str | None = None


def calculate_nba_fatigue(
    team: str,
    recent_games: list[Any],
    target_date: dt.date,
    current_home_team: str,
    is_home: bool,
) -> FatigueProfile:
    """Calculate rest and travel fatigue from the team's most recent game."""

    if not recent_games:
        return FatigueProfile(team, 2, False, 0.0, 0.0, f"WARNING: missing fatigue history for {team}")
    last_game = recent_games[0]
    rest_days = max(0, (target_date - last_game.date).days)
    back_to_back = rest_days <= 1
    last_location_team = team if last_game.is_home else last_game.opponent
    current_location_team = team if is_home else current_home_team
    distance_km = distance_between_teams(last_location_team, current_location_team)
    travel_penalty = clamp(distance_km / 1400.0, 0.0, 4.0)
    fatigue_score = clamp((2.0 - rest_days) * 1.2, -2.0, 3.0) + travel_penalty + (2.2 if back_to_back else 0.0)
    return FatigueProfile(
        team=team,
        rest_days=rest_days,
        back_to_back=back_to_back,
        travel_penalty=travel_penalty,
        fatigue_score=fatigue_score,
    )


def rest_advantage(home: FatigueProfile, away: FatigueProfile) -> float:
    """Positive values favor the home team on rest and travel."""

    return clamp(away.fatigue_score - home.fatigue_score, -8.0, 8.0)


def distance_between_teams(left_team: str, right_team: str) -> float:
    left = team_coords(left_team)
    right = team_coords(right_team)
    if left is None or right is None:
        return 0.0
    return haversine_km(left[0], left[1], right[0], right[1])


def team_coords(team: str) -> tuple[float, float] | None:
    for key, value in NBA_TEAM_COORDS.items():
        if names_match(key, team):
            return value
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

