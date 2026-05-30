from __future__ import annotations

import argparse
import sys
from argparse import Namespace

from core.daily_predictions import generate_daily_predictions
from core.metrics import evaluate_sport
from core.prediction_result import PredictionResult
from core.result_updater import update_results
from core.utils import configure_logging
from sports.football.football_model import FootballPredictor
from sports.nba.nba_model import NBAPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Sports Predictor")
    parser.add_argument("--sport", choices=["nba", "football"], help="Sport to predict.")
    parser.add_argument("--date", default="tomorrow", help="today, tomorrow, or YYYY-MM-DD.")
    parser.add_argument("--home", nargs="*", default="", help="Manual prediction home team.")
    parser.add_argument("--away", nargs="*", default="", help="Manual prediction away team.")
    parser.add_argument("--mode", default="WORLD_CUP", help="Football mode/stage.")
    parser.add_argument("--live", action="store_true", help="Run live schedule prediction.")
    parser.add_argument("--backtest", action="store_true", help="Run sport backtest.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved backtest results.")
    parser.add_argument("--injuries", action="store_true", help="Print NBA injury impact.")
    parser.add_argument("--season", default="2025-26", help="NBA season for backtest/history refresh.")
    parser.add_argument("--limit", type=int, default=100, help="Backtest/history limit.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser


def cli_requested(argv: list[str]) -> bool:
    cli_flags = {"--sport", "--date", "--home", "--away", "--mode", "--live", "--backtest", "--evaluate", "--injuries"}
    return any(arg in cli_flags or arg.startswith("--sport=") or arg.startswith("--date=") for arg in argv)


def run_cli(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    args.home = " ".join(args.home).strip() if isinstance(args.home, list) else str(args.home or "").strip()
    args.away = " ".join(args.away).strip() if isinstance(args.away, list) else str(args.away or "").strip()
    if args.live and not args.sport:
        nba_results = NBAPredictor().predict(
            Namespace(**{**vars(args), "sport": "nba", "home": "", "away": ""})
        )
        football_results = FootballPredictor().predict_live(
            Namespace(**{**vars(args), "sport": "football", "home": "", "away": ""})
        )
        print_predictions(nba_results + football_results)
        return 0
    if not args.sport:
        print("Please provide --sport nba or --sport football")
        return 2
    predictor = NBAPredictor() if args.sport == "nba" else FootballPredictor()
    if args.backtest:
        summary = predictor.backtest(args)
        print_summary(summary)
        return 0
    if args.evaluate:
        print_summary(evaluate_sport(args.sport))
        return 0
    if args.sport == "football" and not (args.home and args.away):
        results = predictor.predict_live(args)
    else:
        results = predictor.predict(args)
    print_predictions(results)
    return 0


def print_summary(summary: dict[str, object]) -> None:
    for key, value in summary.items():
        print(f"{key}: {value}")


def print_predictions(results: list[PredictionResult]) -> None:
    if not results:
        return
    for result in results:
        print("=" * 72)
        print(f"{result.sport.upper()} | {result.home_team} vs {result.away_team} | {result.prediction_date}")
        print(f"Predicted result: {result.predicted_winner}")
        print(f"Home win probability: {format_probability(result.win_probability_home)}")
        if result.draw_probability is not None:
            print(f"Draw probability: {format_probability(result.draw_probability)}")
        print(f"Away win probability: {format_probability(result.win_probability_away)}")
        print(f"Predicted score: {result.predicted_score}")
        print(f"Confidence: {result.confidence}")
        print(f"Data source: {result.data_source}")
        print("Key factors:")
        for item in result.key_factors[:6]:
            print(f"- {item}")
        print("Risk factors:")
        for item in result.risk_factors[:6]:
            print(f"- {item}")


def format_probability(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if "--daily" in sys.argv:
        package = generate_daily_predictions()
        print("Daily prediction generation complete")
        print(f"predictions={len(package.predictions)}")
        print(f"daily_predictions_csv={package.csv_path}")
        print(f"daily_predictions_txt={package.txt_path}")
        print(f"short_script={package.short_script_path}")
        print(f"social_posts={package.social_posts_path}")
    elif "--update-results" in sys.argv:
        result = update_results()
        print("Result update complete")
        print(f"settled={result.get('settled', 0)} pending={result.get('pending', 0)} updated={result.get('updated', 0)}")
        print(f"performance_report={result.get('performance_report')}")
        print(f"recap={result.get('recap')}")
    elif cli_requested(sys.argv[1:]):
        raise SystemExit(run_cli(build_parser().parse_args()))
    else:
        from ui.streamlit_app import main

        main()
