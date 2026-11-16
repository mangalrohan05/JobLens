import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import app.app as main_app

db = main_app.get_db()
skills = main_app.load_all_taxonomy_skills()
main_app.render_profile_page(db, skills)
