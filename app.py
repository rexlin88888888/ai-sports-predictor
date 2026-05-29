from __future__ import annotations

import sys

from core.result_updater import update_results
from ui.streamlit_app import main


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if "--update-results" in sys.argv:
        result = update_results()
        print("Result update complete")
        print(f"settled={result.get('settled', 0)} pending={result.get('pending', 0)} updated={result.get('updated', 0)}")
        print(f"performance_report={result.get('performance_report')}")
        print(f"recap={result.get('recap')}")
    else:
        main()
