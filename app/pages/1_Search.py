import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import app.app as main_app

index, bm25_ranker, baseline_ranker, matcher = main_app.load_search_engine()
db = main_app.get_db()
main_app.render_search_page(index, bm25_ranker, baseline_ranker, db)
