"""Domain-aware tokenizer with protected technical tokens, stopword removal, and stemming."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple
import yaml
from nltk.stem import PorterStemmer

# Standard English stopwords (concise, high-frequency functional words)
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Regex to match protected punctuation-laden tokens first, then regular alphanumeric words
TOKEN_PATTERN = re.compile(
    r"""(?x)
    (?:c\+\+|c\#|\.net|node\.js|vue\.js|next\.js|nuxt\.js|react\.js|d3\.js|chart\.js|ci\/cd|tcp\/ip|wi-fi)
    | [a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*
    """
)


class JobTokenizer:
    """Domain-specific tokenizer for IR indexing and query processing."""

    DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent / "skills" / "taxonomy.yaml"

    def __init__(
        self,
        taxonomy_path: Optional[str | Path] = None,
        stem: bool = True,
        remove_stopwords: bool = True,
    ) -> None:
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        self.protected_tokens: Set[str] = set()

        # Specific mandated protected tokens
        self.mandated_protected = {
            "c++", "c#", ".net", "node.js", "ci/cd", "pandas", "kubernetes", "aws", "gcp", "azure", "sql"
        }
        self.protected_tokens.update(self.mandated_protected)
        self.stem_cache: Dict[str, str] = {}

        # Load taxonomy skills to bypass stemming
        tax_path = Path(taxonomy_path) if taxonomy_path else self.DEFAULT_TAXONOMY_PATH
        if tax_path.exists():
            self._load_taxonomy_protection(tax_path)

    def _load_taxonomy_protection(self, path: Path) -> None:
        """Load taxonomy skills and aliases so they bypass stemming."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for category, skills in data.items():
                for item in skills:
                    canonical = item["canonical"].strip().lower()
                    # Add individual words of canonical skill
                    for word in canonical.split():
                        self.protected_tokens.add(word)
                    for alias in item.get("aliases", []):
                        if alias and alias.strip():
                            for word in alias.strip().lower().split():
                                self.protected_tokens.add(word)
        except Exception:
            pass

    def tokenize_with_positions(self, text: str | None) -> List[Tuple[str, int]]:
        """Tokenize text into (token, position) pairs.

        Args:
            text: Input string.

        Returns:
            List of (normalized_token, token_index) tuples.
        """
        if not text or not isinstance(text, str):
            return []

        tokens_with_pos: List[Tuple[str, int]] = []
        pos = 0

        for match in TOKEN_PATTERN.finditer(text):
            raw_token = match.group(0).lower().strip()
            if not raw_token:
                continue

            # Strip leading/trailing punctuation except for protected tokens
            if raw_token not in self.mandated_protected:
                clean_tok = raw_token.strip(".,;:!?()[]{}\"'`~")
            else:
                clean_tok = raw_token

            if not clean_tok:
                continue

            # Stopword removal
            if self.remove_stopwords and clean_tok in ENGLISH_STOPWORDS:
                continue

            # Stemming (bypass for protected taxonomy skills and symbols, with memoization cache)
            if self.stem and clean_tok not in self.protected_tokens:
                if clean_tok in self.stem_cache:
                    final_tok = self.stem_cache[clean_tok]
                else:
                    final_tok = self.stemmer.stem(clean_tok)
                    self.stem_cache[clean_tok] = final_tok
            else:
                final_tok = clean_tok

            tokens_with_pos.append((final_tok, pos))
            pos += 1

        return tokens_with_pos

    def tokenize(self, text: str | None) -> List[str]:
        """Convenience method returning list of processed token strings."""
        return [tok for tok, _ in self.tokenize_with_positions(text)]
