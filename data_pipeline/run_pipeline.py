from __future__ import annotations

import argparse

from .common import LOGGER, configure_pipeline_logging
from .db import initialize_database, merge_duplicate_matches
from .fetch_elo import fetch_elo_ratings
from .fetch_espn import fetch_live_matches
from .fetch_schedule import fetch_openfootball_schedule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="World Cup data pipeline")
    parser.add_argument("--full", action="store_true", help="Update schedule, live matches and Elo ratings.")
    parser.add_argument("--live", action="store_true", help="Update ESPN live scores only.")
    parser.add_argument("--elo", action="store_true", help="Update Elo ratings only.")
    parser.add_argument("--days-back", type=int, default=1)
    parser.add_argument("--days-forward", type=int, default=7)
    return parser


def main() -> int:
    configure_pipeline_logging()
    initialize_database()
    args = build_parser().parse_args()
    if args.elo:
        rows = fetch_elo_ratings()
        print(f"elo_rows={len(rows)}")
        return 0
    if args.live:
        rows = fetch_live_matches(args.days_back, args.days_forward)
        merge_report = merge_duplicate_matches()
        print(f"live_matches={len(rows)} duplicate_groups_after={merge_report['duplicate_groups_after']}")
        return 0
    if args.full or not (args.live or args.elo):
        schedule = fetch_openfootball_schedule()
        live = fetch_live_matches(args.days_back, args.days_forward)
        elo = fetch_elo_ratings()
        merge_report = merge_duplicate_matches()
        LOGGER.info("full pipeline complete schedule=%s live=%s elo=%s", len(schedule), len(live), len(elo))
        print(
            f"schedule_matches={len(schedule)} live_matches={len(live)} elo_rows={len(elo)} "
            f"duplicate_groups_before={merge_report['duplicate_groups_before']} "
            f"duplicate_groups_after={merge_report['duplicate_groups_after']}"
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
