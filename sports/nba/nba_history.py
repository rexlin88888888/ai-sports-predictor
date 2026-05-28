# Required third-party libraries:
#   pip install nba_api pandas requests
#
# Example:
#   streamlit run app.py

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging

import pandas as pd

try:
    from .nba_data import CompletedGame, NBADataClient, ScheduledGame, TeamGame, leaguegamefinder
    from .nba_scoring_model import build_team_metrics, predict_game
    from .nba_utils import NBA_BACKTEST_RESULTS_CSV, NBA_GAMES_CSV, configure_logging, ensure_data_dir, mean, names_match, pct, season_from_date
except ImportError:
    from nba_data import CompletedGame, NBADataClient, ScheduledGame, TeamGame, leaguegamefinder
    from nba_scoring_model import build_team_metrics, predict_game
    from nba_utils import NBA_BACKTEST_RESULTS_CSV, NBA_GAMES_CSV, configure_logging, ensure_data_dir, mean, names_match, pct, season_from_date


LOGGER = logging.getLogger("ai_sports_predictor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest NBA win and score predictions.")
    parser.add_argument("--season", default="2025-26", help="NBA season, for example 2025-26.")
    parser.add_argument("--limit", type=int, default=120, help="Maximum completed games to test.")
    parser.add_argument("--verbose", action="store_true", help="Print debug logs.")
    return parser.parse_args()


def load_completed_games_for_backtest(client: NBADataClient, season: str) -> list[CompletedGame]:
    cached = load_completed_games_from_cache(season)
    if cached:
        LOGGER.info("Using cached NBA completed games from %s: %s game(s).", NBA_GAMES_CSV, len(cached))
        return cached

    leaguewide = fetch_completed_games_leaguewide(season)
    if leaguewide:
        save_completed_games_cache(leaguewide)
        return leaguewide

    teams = client.team_id_to_name()
    if not teams:
        LOGGER.warning("NBA team metadata unavailable; cannot fetch backtest games from API.")
        return []
    records: dict[str, dict[str, object]] = {}
    for team_id, team_name in teams.items():
        # Fetching every team is slower, but it is robust when league-wide finder is restricted.
        team_games = client._get_team_games_from_api(team_name, season, limit=100)
        for game in team_games:
            bucket = records.setdefault(
                game.game_id or f"{game.date}-{game.team}-{game.opponent}",
                {"date": game.date},
            )
            if game.is_home:
                bucket.update(
                    {
                        "home_team": game.team,
                        "away_team": game.opponent,
                        "home_score": game.team_score,
                        "away_score": game.opponent_score,
                    }
                )
            else:
                bucket.update(
                    {
                        "home_team": game.opponent,
                        "away_team": game.team,
                        "home_score": game.opponent_score,
                        "away_score": game.team_score,
                    }
                )
    completed: list[CompletedGame] = []
    for game_id, row in records.items():
        required = {"date", "home_team", "away_team", "home_score", "away_score"}
        if not required <= set(row):
            continue
        completed.append(
            CompletedGame(
                game_id=game_id,
                date=row["date"],  # type: ignore[arg-type]
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
            )
        )
    completed.sort(key=lambda item: item.date)
    save_completed_games_cache(completed)
    return completed


def fetch_completed_games_leaguewide(season: str) -> list[CompletedGame]:
    if leaguegamefinder is None:
        return []
    frames = []
    for season_type in ("Regular Season", "Playoffs"):
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                timeout=30,
            )
            frame = finder.get_data_frames()[0]
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            LOGGER.warning("League-wide NBA gamefinder failed for %s %s: %s", season, season_type, exc)
    if not frames:
        return []
    merged = pd.concat(frames, ignore_index=True)
    completed: list[CompletedGame] = []
    for game_id, group in merged.groupby("GAME_ID"):
        if len(group) < 2:
            continue
        home_rows = group[group["MATCHUP"].astype(str).str.contains(" vs. ", regex=False)]
        away_rows = group[group["MATCHUP"].astype(str).str.contains(" @ ", regex=False)]
        if home_rows.empty or away_rows.empty:
            continue
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]
        try:
            game_date = dt.date.fromisoformat(str(home["GAME_DATE"])[:10])
            completed.append(
                CompletedGame(
                    game_id=str(game_id),
                    date=game_date,
                    home_team=str(home["TEAM_NAME"]),
                    away_team=str(away["TEAM_NAME"]),
                    home_score=int(float(home["PTS"])),
                    away_score=int(float(away["PTS"])),
                )
            )
        except Exception:
            continue
    completed.sort(key=lambda item: item.date)
    LOGGER.info("Fetched league-wide completed NBA games: %s.", len(completed))
    return completed


def load_completed_games_from_cache(season: str) -> list[CompletedGame]:
    if not NBA_GAMES_CSV.exists():
        return []
    games: list[CompletedGame] = []
    try:
        with NBA_GAMES_CSV.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("home_score") or not row.get("away_score"):
                    continue
                try:
                    game_date = dt.date.fromisoformat(str(row["date"])[:10])
                except (KeyError, ValueError):
                    continue
                if season_from_date(game_date) != season:
                    continue
                games.append(
                    CompletedGame(
                        game_id=str(row.get("game_id") or ""),
                        date=game_date,
                        home_team=str(row.get("home_team") or ""),
                        away_team=str(row.get("away_team") or ""),
                        home_score=int(float(row["home_score"])),
                        away_score=int(float(row["away_score"])),
                    )
                )
    except Exception as exc:
        LOGGER.warning("Could not load NBA backtest cache %s: %s", NBA_GAMES_CSV, exc)
        return []
    games.sort(key=lambda item: item.date)
    return games


def save_completed_games_cache(games: list[CompletedGame]) -> None:
    if not games:
        return
    ensure_data_dir()
    with NBA_GAMES_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["game_id", "date", "home_team", "away_team", "home_score", "away_score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for game in games:
            writer.writerow(
                {
                    "game_id": game.game_id,
                    "date": game.date.isoformat(),
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                }
            )


def team_history_from_completed(games: list[CompletedGame], team: str, before_date: dt.date, limit: int = 40) -> list[TeamGame]:
    history: list[TeamGame] = []
    for game in games:
        if game.date >= before_date:
            continue
        if names_match(game.home_team, team):
            history.append(
                TeamGame(game.game_id, game.date, game.home_team, game.away_team, True, game.home_score, game.away_score)
            )
        elif names_match(game.away_team, team):
            history.append(
                TeamGame(game.game_id, game.date, game.away_team, game.home_team, False, game.away_score, game.home_score)
            )
    history.sort(key=lambda item: item.date, reverse=True)
    return history[:limit]


def run_backtest(season: str, limit: int) -> None:
    ensure_data_dir()
    client = NBADataClient()
    completed = load_completed_games_for_backtest(client, season)
    if not completed:
        print("No completed NBA games available for backtest.")
        return
    test_games = completed[-limit:]
    rows: list[dict[str, object]] = []
    correct = 0
    score_errors: list[float] = []
    total_errors: list[float] = []
    for game in test_games:
        home_history = team_history_from_completed(completed, game.home_team, game.date, limit=40)
        away_history = team_history_from_completed(completed, game.away_team, game.date, limit=40)
        home_metrics = build_team_metrics(game.home_team, home_history, game.date, [])
        away_metrics = build_team_metrics(game.away_team, away_history, game.date, [])
        scheduled = ScheduledGame(game.game_id, game.date, "Final", game.home_team, game.away_team)
        prediction = predict_game(scheduled, home_metrics, away_metrics)
        predicted_winner = prediction.predicted_winner
        actual_winner = game.winner
        is_correct = predicted_winner == actual_winner
        correct += int(is_correct)
        score_error = (
            abs(prediction.predicted_home_score - game.home_score)
            + abs(prediction.predicted_away_score - game.away_score)
        ) / 2.0
        total_error = abs(prediction.predicted_total - (game.home_score + game.away_score))
        score_errors.append(score_error)
        total_errors.append(total_error)
        rows.append(
            {
                "date": game.date.isoformat(),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "predicted_winner": predicted_winner,
                "actual_winner": actual_winner,
                "correct": is_correct,
                "predicted_home_score": prediction.predicted_home_score,
                "predicted_away_score": prediction.predicted_away_score,
                "actual_home_score": game.home_score,
                "actual_away_score": game.away_score,
                "score_error": round(score_error, 2),
                "total_points_error": round(total_error, 2),
            }
        )
    save_results(rows)
    accuracy = correct / len(rows) if rows else 0.0
    print(f"Backtest games: {len(rows)}")
    print(f"Win prediction accuracy: {pct(accuracy)}")
    print(f"Average score error: {mean(score_errors):.2f}")
    print(f"Total points error: {mean(total_errors):.2f}")
    print(f"Saved: {NBA_BACKTEST_RESULTS_CSV}")


def save_results(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_data_dir()
    with NBA_BACKTEST_RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    run_backtest(args.season, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
