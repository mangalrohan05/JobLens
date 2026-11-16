"""SQLite persistence layer for candidate profiles, saved opportunities, and search history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobLensDatabase:
    """Thread-safe SQLite storage manager for JobLens user state."""

    DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "joblens.db"

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidate_profiles (
                    user_id TEXT PRIMARY KEY,
                    skills TEXT NOT NULL,
                    years_experience REAL DEFAULT 0.0,
                    preferred_roles TEXT NOT NULL,
                    preferred_locations TEXT NOT NULL,
                    preferred_work_modes TEXT NOT NULL,
                    preferred_cities TEXT DEFAULT '[]',
                    preferred_states TEXT DEFAULT '[]',
                    min_salary REAL,
                    updated_at TEXT NOT NULL
                )
            """)
            # Ensure columns exist if table was previously created with earlier schema
            cursor.execute("PRAGMA table_info(candidate_profiles)")
            columns = [col["name"] for col in cursor.fetchall()]
            if "preferred_cities" not in columns:
                cursor.execute("ALTER TABLE candidate_profiles ADD COLUMN preferred_cities TEXT DEFAULT '[]'")
            if "preferred_states" not in columns:
                cursor.execute("ALTER TABLE candidate_profiles ADD COLUMN preferred_states TEXT DEFAULT '[]'")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS saved_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    title TEXT,
                    company TEXT,
                    location TEXT,
                    match_score REAL,
                    saved_at TEXT NOT NULL,
                    notes TEXT,
                    UNIQUE(user_id, job_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    filters_json TEXT,
                    result_count INTEGER DEFAULT 0,
                    searched_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_profile(
        self,
        user_id: str,
        skills: List[str],
        years_experience: float,
        preferred_roles: List[str],
        preferred_locations: List[str],
        preferred_work_modes: List[str],
        min_salary: Optional[float] = None,
        preferred_cities: Optional[List[str]] = None,
        preferred_states: Optional[List[str]] = None,
    ) -> None:
        """Insert or replace candidate profile."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO candidate_profiles (
                    user_id, skills, years_experience, preferred_roles,
                    preferred_locations, preferred_work_modes, preferred_cities,
                    preferred_states, min_salary, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                json.dumps(skills),
                float(years_experience),
                json.dumps(preferred_roles),
                json.dumps(preferred_locations),
                json.dumps(preferred_work_modes),
                json.dumps(preferred_cities or []),
                json.dumps(preferred_states or []),
                float(min_salary) if min_salary is not None else None,
                now_iso,
            ))
            conn.commit()

    def get_profile(self, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        """Retrieve candidate profile by user_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidate_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            keys = row.keys()
            return {
                "user_id": row["user_id"],
                "skills": json.loads(row["skills"]),
                "years_experience": float(row["years_experience"]),
                "preferred_roles": json.loads(row["preferred_roles"]),
                "preferred_locations": json.loads(row["preferred_locations"]),
                "preferred_work_modes": json.loads(row["preferred_work_modes"]),
                "preferred_cities": json.loads(row["preferred_cities"]) if "preferred_cities" in keys and row["preferred_cities"] else [],
                "preferred_states": json.loads(row["preferred_states"]) if "preferred_states" in keys and row["preferred_states"] else [],
                "min_salary": row["min_salary"],
                "updated_at": row["updated_at"],
            }

    def save_job(
        self,
        user_id: str,
        job_id: str,
        title: str,
        company: Optional[str] = None,
        location: Optional[str] = None,
        match_score: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Save a job posting for a user. Returns True if saved, False if already saved."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO saved_jobs (
                        user_id, job_id, title, company, location, match_score, saved_at, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, job_id, title, company, location, match_score, now_iso, notes or ""
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def remove_saved_job(self, user_id: str, job_id: str) -> bool:
        """Remove a saved job."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM saved_jobs WHERE user_id = ? AND job_id = ?", (user_id, job_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_saved_jobs(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieve list of all saved jobs for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM saved_jobs WHERE user_id = ? ORDER BY saved_at DESC",
                (user_id,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def record_search(
        self,
        user_id: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        result_count: int = 0,
    ) -> None:
        """Log a search query into history."""
        now_iso = datetime.now(timezone.utc).isoformat()
        filters_str = json.dumps(filters) if filters else None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO search_history (
                    user_id, query, filters_json, result_count, searched_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (user_id, query, filters_str, result_count, now_iso))
            conn.commit()

    def get_search_history(self, user_id: str = "default_user", limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent search queries for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM search_history WHERE user_id = ? ORDER BY searched_at DESC LIMIT ?",
                (user_id, limit),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
