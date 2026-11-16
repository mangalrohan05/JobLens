"""Skill taxonomy management and boundary-aware phrase extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml


class SkillExtractor:
    """Extracts standardized skills from job posting text using a curated taxonomy."""

    DEFAULT_TAXONOMY_PATH = Path(__file__).parent / "taxonomy.yaml"

    def __init__(self, taxonomy_path: str | Path | None = None) -> None:
        """Initialize the extractor with a YAML taxonomy file."""
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else self.DEFAULT_TAXONOMY_PATH
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at: {self.taxonomy_path}")

        self.taxonomy: Dict[str, List[Dict[str, any]]] = {}
        self.canonical_to_category: Dict[str, str] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        self._compiled_patterns: List[Tuple[re.Pattern, str, str]] = []

        self._load_and_compile()

    def _load_and_compile(self) -> None:
        """Load taxonomy YAML and precompile regex matching patterns."""
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = yaml.safe_load(f)

        raw_patterns: List[Tuple[str, str, str]] = []

        for category, skill_list in self.taxonomy.items():
            for item in skill_list:
                canonical = item["canonical"].strip().lower()
                self.canonical_to_category[canonical] = category

                aliases = set()
                aliases.add(canonical)
                for alias in item.get("aliases", []):
                    if alias and alias.strip():
                        aliases.add(alias.strip().lower())

                for alias in aliases:
                    self.alias_to_canonical[alias] = canonical
                    pattern = self._build_pattern_for_alias(alias)
                    raw_patterns.append((alias, pattern, canonical))

        # Sort patterns by length descending so longer phrases match first
        raw_patterns.sort(key=lambda x: len(x[0]), reverse=True)

        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), canonical, alias.lower())
            for alias, pattern, canonical in raw_patterns
        ]

    def _build_pattern_for_alias(self, alias: str) -> str:
        """Construct a regex pattern with appropriate boundary protection without variable-width lookbehinds."""
        if alias == "c":
            return r"(?:\bc\s+(?:programming|language|developer|engineer|code)\b|\b(?:ansi\s+c|c\/c\+\+)\b)"
        if alias == "r":
            return r"(?:\br\s+(?:programming|language|package|packages|script|stats)\b|\b(?:rstats|cran)\b)"
        if alias == "go":
            return r"(?:\bgo\s+(?:programming|language|developer|engineer|backend|routine|routines|code)\b|\bgolang\b)"
        if alias == "sh":
            return r"(?:\bsh\s+(?:script|scripting|command|shell)\b)"
        if alias == "cv":
            return r"(?:\bcv\s+(?:engineer|algorithm|researcher|model|pipeline)\b|\bcomputer\s+vision\b)"
        if alias == "bi":
            return r"(?:\bbi\s+(?:tool|tools|dashboard|developer|analyst|reporting|analytics)\b|\bbusiness\s+intelligence\b)"
        if alias == "eq":
            return r"(?:\beq\s+(?:or\s+emotional\s+intelligence)\b)"

        escaped = re.escape(alias)

        if alias.startswith((".", "#", "+")):
            left = r"(?:^|[^a-zA-Z0-9])"
        else:
            left = r"\b"

        if alias.endswith(("+", "#")):
            right = r"(?=$|[^a-zA-Z0-9\+#])"
        else:
            right = r"\b"

        return f"{left}{escaped}{right}"

    def extract_skills(self, text: str | None) -> List[str]:
        """Extract all recognized canonical skills from text in deterministic order.

        Uses fast C-level substring pre-filtering before regex matching for high performance.
        """
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        matched_canonicals: Set[str] = set()

        for pattern, canonical, alias_lower in self._compiled_patterns:
            if canonical in matched_canonicals:
                continue
            # Fast substring pre-check in C
            if alias_lower not in text_lower:
                continue
            if pattern.search(text):
                matched_canonicals.add(canonical)

        return sorted(matched_canonicals)

    def get_category(self, canonical_skill: str) -> str:
        """Return the category of a canonical skill."""
        return self.canonical_to_category.get(canonical_skill.lower(), "other")

    def get_stats(self) -> Dict[str, any]:
        """Return summary statistics of the taxonomy."""
        category_counts: Dict[str, int] = {}
        for category, skills in self.taxonomy.items():
            category_counts[category] = len(skills)
        return {
            "total_canonical_skills": len(self.canonical_to_category),
            "total_aliases": len(self.alias_to_canonical),
            "categories": category_counts,
        }
