from __future__ import annotations

try:
    from .sports.nba.injury_data import InjuryDataClient, InjuryRecord, TeamInjuryImpact
except ImportError:
    from sports.nba.injury_data import InjuryDataClient, InjuryRecord, TeamInjuryImpact


__all__ = ["InjuryDataClient", "InjuryRecord", "TeamInjuryImpact"]

