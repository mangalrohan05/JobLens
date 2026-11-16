"""Multi-field inverted index with positional postings, collection statistics, and JSON/NPZ persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from joblens.ir.tokenizer import JobTokenizer


class InvertedIndex:
    """Multi-field inverted index with positional postings and collection statistics."""

    FIELDS = ["title", "skills", "description"]

    def __init__(self, tokenizer: Optional[JobTokenizer] = None) -> None:
        self.tokenizer = tokenizer or JobTokenizer()
        self.N: int = 0  # Total indexed documents
        # postings[field][term] = {doc_id: (tf, [positions])}
        self.postings: Dict[str, Dict[str, Dict[str, Tuple[int, List[int]]]]] = {
            f: {} for f in self.FIELDS
        }
        # doc_lengths[field][doc_id] = token_count
        self.doc_lengths: Dict[str, Dict[str, int]] = {f: {} for f in self.FIELDS}
        # avg_doc_lengths[field] = float
        self.avg_doc_lengths: Dict[str, float] = {f: 0.0 for f in self.FIELDS}
        # doc_freq[term] = count of unique documents containing term in ANY field
        self.doc_freq: Dict[str, int] = {}
        # doc_metadata[doc_id] = dict of job attributes
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
        # Set of all indexed doc_ids
        self.doc_ids: List[str] = []

    def add_document(
        self,
        doc_id: str,
        title: str,
        skills: List[str] | str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a single document across title, skills, and description fields."""
        doc_id_str = str(doc_id)
        if doc_id_str in self.doc_metadata:
            # Already indexed, skip or update
            return

        self.doc_ids.append(doc_id_str)
        self.N += 1

        # Store metadata
        meta = metadata.copy() if metadata else {}
        meta["job_id"] = doc_id_str
        meta["title"] = title
        if isinstance(skills, (list, set, tuple)) or hasattr(skills, "__iter__") and not isinstance(skills, str):
            skills_list = [str(s).strip() for s in skills if str(s).strip()]
        elif isinstance(skills, str):
            skills_list = [s.strip() for s in skills.split(",") if s.strip()]
        else:
            skills_list = []
        meta["skills"] = skills_list
        meta["description"] = description
        self.doc_metadata[doc_id_str] = meta

        # Process each field
        skills_text = " ".join(skills_list)
        field_texts = {
            "title": title or "",
            "skills": skills_text,
            "description": description or "",
        }

        doc_terms_all_fields: Set[str] = set()

        for field, text in field_texts.items():
            tokens_with_pos = self.tokenizer.tokenize_with_positions(text)
            self.doc_lengths[field][doc_id_str] = len(tokens_with_pos)

            # Build term frequency and positions for this field
            term_positions: Dict[str, List[int]] = {}
            for term, pos in tokens_with_pos:
                doc_terms_all_fields.add(term)
                if term not in term_positions:
                    term_positions[term] = []
                term_positions[term].append(pos)

            # Add to field postings
            field_post = self.postings[field]
            for term, positions in term_positions.items():
                if term not in field_post:
                    field_post[term] = {}
                field_post[term][doc_id_str] = (len(positions), positions)

        # Update global collection document frequency for terms seen in this document
        for term in doc_terms_all_fields:
            self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def finalize(self) -> None:
        """Compute collection averages after indexing completes."""
        if self.N > 0:
            for field in self.FIELDS:
                total_len = sum(self.doc_lengths[field].values())
                self.avg_doc_lengths[field] = total_len / self.N
        else:
            for field in self.FIELDS:
                self.avg_doc_lengths[field] = 0.0

    def get_field_postings(self, field: str, term: str) -> Dict[str, Tuple[int, List[int]]]:
        """Return postings dict {doc_id: (tf, positions)} for a term in a given field."""
        return self.postings.get(field, {}).get(term, {})

    def get_document_frequency(self, term: str) -> int:
        """Return document frequency of term across all fields."""
        return self.doc_freq.get(term, 0)

    def save(self, output_dir: str | Path) -> None:
        """Persist index to disk as JSON/NPZ (no pickle)."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        self.finalize()

        # 1. Save metadata, lengths, and collection statistics
        meta_payload = {
            "N": self.N,
            "FIELDS": self.FIELDS,
            "avg_doc_lengths": self.avg_doc_lengths,
            "doc_lengths": self.doc_lengths,
            "doc_freq": self.doc_freq,
            "doc_ids": self.doc_ids,
            "doc_metadata": self.doc_metadata,
        }
        with open(out_path / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        # 2. Save postings as structured JSON
        # Structure: {field: {term: {doc_id: [tf, [positions]]}}}
        serializable_postings = {}
        for field, term_map in self.postings.items():
            serializable_postings[field] = {}
            for term, doc_map in term_map.items():
                serializable_postings[field][term] = {
                    doc_id: [tf, pos] for doc_id, (tf, pos) in doc_map.items()
                }

        with open(out_path / "index_postings.json", "w", encoding="utf-8") as f:
            json.dump(serializable_postings, f)

    @classmethod
    def load(cls, input_dir: str | Path, tokenizer: Optional[JobTokenizer] = None) -> InvertedIndex:
        """Load index from JSON/NPZ files."""
        in_path = Path(input_dir)
        if not (in_path / "index_meta.json").exists():
            raise FileNotFoundError(f"Index metadata not found in: {in_path}")

        index = cls(tokenizer=tokenizer)

        with open(in_path / "index_meta.json", "r", encoding="utf-8") as f:
            meta_payload = json.load(f)

        index.N = meta_payload["N"]
        index.avg_doc_lengths = meta_payload["avg_doc_lengths"]
        index.doc_lengths = meta_payload["doc_lengths"]
        index.doc_freq = meta_payload["doc_freq"]
        index.doc_ids = meta_payload["doc_ids"]
        index.doc_metadata = meta_payload["doc_metadata"]

        with open(in_path / "index_postings.json", "r", encoding="utf-8") as f:
            raw_postings = json.load(f)

        for field, term_map in raw_postings.items():
            index.postings[field] = {}
            for term, doc_map in term_map.items():
                index.postings[field][term] = {
                    doc_id: (int(tf), list(pos)) for doc_id, (tf, pos) in doc_map.items()
                }

        return index
