from __future__ import annotations

import sys

from core.daily_predictions import generate_daily_predictions
from core.result_updater import update_results


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
    else:
        from ui.streamlit_app import main

        main()
