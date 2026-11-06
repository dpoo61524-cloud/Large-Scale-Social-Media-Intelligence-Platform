"""LIAR Benchmark Dataset Loader.

Supports ingestion of standard LIAR TSV and JSON datasets, label mapping
(6-class fine-grained and binary targets), and embedded development fixture
fallback for deterministic offline testing.

Important Scope Note:
- The embedded sample of 24 statements is an offline development test fixture,
  not the official LIAR benchmark (which contains 12,836 statements).
- Official benchmark evaluation requires an external LIAR dataset path supplied
  by the user; data/raw/ is never read, modified, or accessed.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Standard LIAR benchmark class definitions
FINE_GRAINED_LABEL_TO_ID: Dict[str, int] = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

ID_TO_FINE_GRAINED_LABEL: Dict[int, str] = {
    v: k for k, v in FINE_GRAINED_LABEL_TO_ID.items()
}

BINARY_LABEL_TO_ID: Dict[str, int] = {
    "fake": 0,
    "real": 1,
}

ID_TO_BINARY_LABEL: Dict[int, str] = {
    0: "fake",
    1: "real",
}

# Embedded representative benchmark sample across all 6 veracity tiers
EMBEDDED_BENCHMARK_SAMPLE: List[Dict[str, Any]] = [
    # Pants-on-fire (completely baseless/conspiracy)
    {
        "statement": "The moon is completely hollow and secretly housing extraterrestrial military facilities.",
        "label": "pants-fire",
        "speaker": "Anonymous Blog",
        "context": "social-forum-post",
    },
    {
        "statement": "Microscopic electronic tracking microchips are secretly dissolved in all municipal tap water.",
        "label": "pants-fire",
        "speaker": "Online Viral Video",
        "context": "video-stream",
    },
    {
        "statement": "Drinking industrial chlorine bleach guarantees 100% immunity against all bacterial and viral infections.",
        "label": "pants-fire",
        "speaker": "Alternative Remedy Forum",
        "context": "forum-thread",
    },
    {
        "statement": "Global satellite organizations admit the earth has no gravitational pull and is entirely flat.",
        "label": "pants-fire",
        "speaker": "Conspiracy Channel",
        "context": "online-broadcast",
    },
    # False (demonstrably untrue factual claims)
    {
        "statement": "National unemployment hit a record high of ninety percent across the entire workforce last month.",
        "label": "false",
        "speaker": "Political Commentary",
        "context": "press-release",
    },
    {
        "statement": "Renewable solar energy plants consume ten times more fossil fuels than they ever produce.",
        "label": "false",
        "speaker": "Energy Lobbyist",
        "context": "radio-interview",
    },
    {
        "statement": "The national debt was completely liquidated and reduced to zero dollars in the first quarter of 2024.",
        "label": "false",
        "speaker": "Financial Blogger",
        "context": "newsletter",
    },
    {
        "statement": "The national electric grid has had zero operating turbines functioning for three consecutive months.",
        "label": "false",
        "speaker": "Industry Critic",
        "context": "op-ed-column",
    },
    # Barely-true (heavily distorted or out-of-context claim)
    {
        "statement": "Domestic fuel prices skyrocketed solely due to a single administrative regulation signed yesterday.",
        "label": "barely-true",
        "speaker": "Opposition Spokesperson",
        "context": "tv-debate",
    },
    {
        "statement": "Violent crime rates increased by over five hundred percent in every single metropolitan area.",
        "label": "barely-true",
        "speaker": "Campaign Speech",
        "context": "rally-address",
    },
    {
        "statement": "Over eighty percent of national tax revenues are sent directly as foreign cash handouts.",
        "label": "barely-true",
        "speaker": "Talk Show Guest",
        "context": "podcast-episode",
    },
    {
        "statement": "Every new vehicle produced today is legally prohibited from using internal combustion engines.",
        "label": "barely-true",
        "speaker": "Auto Enthusiast Site",
        "context": "editorial-note",
    },
    # Half-true (partially accurate but omits key context)
    {
        "statement": "The new infrastructure bill created two million construction jobs in its first active fiscal year.",
        "label": "half-true",
        "speaker": "Government Representative",
        "context": "press-briefing",
    },
    {
        "statement": "Consumer electronics prices dropped while supply chain transit costs rose in foreign markets.",
        "label": "half-true",
        "speaker": "Market Analyst",
        "context": "financial-report",
    },
    {
        "statement": "Electric vehicle purchases slowed in several regional markets due to high charging rates.",
        "label": "half-true",
        "speaker": "Automotive Journalist",
        "context": "industry-article",
    },
    {
        "statement": "Corporate tax revenues rose overall despite a reduction in top marginal corporate tax rates.",
        "label": "half-true",
        "speaker": "Economic Researcher",
        "context": "policy-paper",
    },
    # Mostly-true (accurate overall with minor qualifications)
    {
        "statement": "Commercial solar installations expanded by over twenty-two percent across suburban sectors last year.",
        "label": "mostly-true",
        "speaker": "Clean Energy Council",
        "context": "annual-review",
    },
    {
        "statement": "Independent audits verified that small business grants went primarily to local family-owned retail stores.",
        "label": "mostly-true",
        "speaker": "Auditor General",
        "context": "congressional-hearing",
    },
    {
        "statement": "Public healthcare records show that standard childhood immunizations lower viral transmission rates significantly.",
        "label": "mostly-true",
        "speaker": "Medical Association",
        "context": "journal-publication",
    },
    {
        "statement": "High-speed broadband network access reached ninety-four percent of newly developed municipal areas.",
        "label": "mostly-true",
        "speaker": "Telecom Director",
        "context": "infrastructure-update",
    },
    # True (completely verified factual statements)
    {
        "statement": "The central bank raised the target federal funds interest rate by twenty-five basis points at today's meeting.",
        "label": "true",
        "speaker": "Central Bank Chair",
        "context": "monetary-policy-statement",
    },
    {
        "statement": "NASA's James Webb Space Telescope detected primeval galaxies formed within four hundred million years of the Big Bang.",
        "label": "true",
        "speaker": "Astrophysics Institute",
        "context": "scientific-release",
    },
    {
        "statement": "National gross domestic product expanded at an annualized rate of 2.8 percent in the third quarter.",
        "label": "true",
        "speaker": "Bureau of Economic Analysis",
        "context": "statistical-release",
    },
    {
        "statement": "Water molecules consist of two hydrogen atoms bonded to one oxygen atom under standard thermodynamic conditions.",
        "label": "true",
        "speaker": "Science Educator",
        "context": "academic-textbook",
    },
]


class LiarDatasetLoader:
    """Loader and preprocessor for LIAR benchmark and veracity datasets."""

    def __init__(self, data_path: Optional[str | Path] = None) -> None:
        """Initialize dataset loader with optional path to dataset file."""
        self.data_path = Path(data_path) if data_path else None

    @staticmethod
    def get_fine_grained_labels() -> List[str]:
        """Return list of original 6-class labels."""
        return list(FINE_GRAINED_LABEL_TO_ID.keys())

    @staticmethod
    def get_binary_labels() -> List[str]:
        """Return list of binary veracity labels."""
        return ["fake", "real"]

    @staticmethod
    def normalize_label(label: str) -> str:
        """Normalize raw string label to standard lowercase hyphenated key."""
        cleaned = str(label).strip().lower().replace("_", "-").replace(" ", "-")
        # Handle variations
        alias_map = {
            "pants-on-fire": "pants-fire",
            "pantsfire": "pants-fire",
            "barely": "barely-true",
            "mostly": "mostly-true",
            "half": "half-true",
        }
        return alias_map.get(cleaned, cleaned)

    def map_to_binary(
        self, label: str, uncertain_policy: str = "exclude"
    ) -> Optional[int]:
        """Map fine-grained label to binary target (0=fake, 1=real).

        Parameters
        ----------
        label : str
            Fine-grained label name.
        uncertain_policy : str
            Policy for 'half-true' label:
            - 'exclude': returns None (skipped in strict binary mode)
            - 'fake': maps to 0
            - 'real': maps to 1

        Returns
        -------
        Optional[int]
            0 for fake, 1 for real, or None if excluded.
        """
        norm = self.normalize_label(label)
        if norm in ("pants-fire", "false", "barely-true"):
            return 0
        if norm in ("mostly-true", "true"):
            return 1
        if norm == "half-true":
            if uncertain_policy == "fake":
                return 0
            if uncertain_policy == "real":
                return 1
            return None  # 'exclude'
        # Unknown label
        return None

    def map_to_fine_grained(self, label: str) -> Optional[int]:
        """Map fine-grained label name to ID (0 to 5)."""
        norm = self.normalize_label(label)
        return FINE_GRAINED_LABEL_TO_ID.get(norm)

    def load_benchmark_sample(self) -> List[Dict[str, Any]]:
        """Return built-in sample benchmark dataset for offline testing."""
        return [dict(item) for item in EMBEDDED_BENCHMARK_SAMPLE]

    def load_from_tsv(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Load LIAR dataset from standard TSV file.

        Supports standard 14-column LIAR TSV or headered TSV format.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"LIAR TSV file not found: {path}")

        records: List[Dict[str, Any]] = []
        with open(path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            first_row = next(reader, None)
            if first_row is None:
                return records

            # Check if header row exists
            headers_lower = [h.strip().lower() for h in first_row]
            if "statement" in headers_lower and "label" in headers_lower:
                stmt_idx = headers_lower.index("statement")
                lbl_idx = headers_lower.index("label")
                spk_idx = headers_lower.index("speaker") if "speaker" in headers_lower else None
                ctx_idx = headers_lower.index("context") if "context" in headers_lower else None

                for row in reader:
                    if len(row) <= max(stmt_idx, lbl_idx):
                        continue
                    records.append({
                        "statement": row[stmt_idx].strip(),
                        "label": self.normalize_label(row[lbl_idx]),
                        "speaker": row[spk_idx].strip() if spk_idx is not None and len(row) > spk_idx else "unknown",
                        "context": row[ctx_idx].strip() if ctx_idx is not None and len(row) > ctx_idx else "unknown",
                    })
            else:
                # Standard LIAR benchmark TSV format (unheadered 14 columns):
                # col 0: id, col 1: label, col 2: statement, col 3: subject, col 4: speaker, col 13: context
                def parse_standard_row(row_items: List[str]) -> Optional[Dict[str, Any]]:
                    if len(row_items) >= 3:
                        # Col 1 is label, Col 2 is statement
                        lbl = row_items[1].strip()
                        stmt = row_items[2].strip()
                        spk = row_items[4].strip() if len(row_items) > 4 else "unknown"
                        ctx = row_items[13].strip() if len(row_items) > 13 else "unknown"
                        return {
                            "statement": stmt,
                            "label": self.normalize_label(lbl),
                            "speaker": spk,
                            "context": ctx,
                        }
                    return None

                parsed = parse_standard_row(first_row)
                if parsed:
                    records.append(parsed)
                for row in reader:
                    parsed = parse_standard_row(row)
                    if parsed:
                        records.append(parsed)

        return records

    def load_from_json(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Load records from a JSON file (list of dicts or dict with records key)."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"LIAR JSON file not found: {path}")

        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            raw_items = data.get("records", data.get("data", data.get("statements", [])))
        elif isinstance(data, list):
            raw_items = data
        else:
            raw_items = []

        records: List[Dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, dict) and "statement" in item and "label" in item:
                records.append({
                    "statement": str(item["statement"]).strip(),
                    "label": self.normalize_label(str(item["label"])),
                    "speaker": str(item.get("speaker", "unknown")).strip(),
                    "context": str(item.get("context", "unknown")).strip(),
                })
        return records

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load records from configured file path or fallback to embedded benchmark sample."""
        if self.data_path and self.data_path.exists():
            if self.data_path.suffix.lower() == ".json":
                return self.load_from_json(self.data_path)
            if self.data_path.suffix.lower() in (".tsv", ".csv", ".txt"):
                return self.load_from_tsv(self.data_path)
        return self.load_benchmark_sample()

    def prepare_data(
        self,
        binary: bool = True,
        uncertain_policy: str = "exclude",
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        """Prepare parallel lists of statements, labels (int), and metadata.

        Parameters
        ----------
        binary : bool
            If True, maps to 0 (fake) and 1 (real). If False, maps to 0..5 fine-grained.
        uncertain_policy : str
            Policy for handling 'half-true' in binary mode ('exclude', 'fake', 'real').
        records : Optional[List[Dict[str, Any]]]
            If provided, uses these records; otherwise loads from configured path or fallback.

        Returns
        -------
        Tuple[List[str], List[int], List[Dict[str, Any]]]
            Parallel lists of texts, integer labels, and record metadata dicts.
        """
        raw_records = records if records is not None else self.load_dataset()
        statements: List[str] = []
        labels: List[int] = []
        metadata: List[Dict[str, Any]] = []

        for rec in raw_records:
            stmt = rec.get("statement", "").strip()
            raw_lbl = rec.get("label", "")
            if not stmt:
                continue

            if binary:
                mapped_lbl = self.map_to_binary(raw_lbl, uncertain_policy=uncertain_policy)
                if mapped_lbl is None:
                    continue  # skipped by policy
                statements.append(stmt)
                labels.append(mapped_lbl)
                metadata.append({
                    "speaker": rec.get("speaker", "unknown"),
                    "context": rec.get("context", "unknown"),
                    "original_label": raw_lbl,
                    "target_type": "binary",
                })
            else:
                fine_lbl = self.map_to_fine_grained(raw_lbl)
                if fine_lbl is None:
                    continue
                statements.append(stmt)
                labels.append(fine_lbl)
                metadata.append({
                    "speaker": rec.get("speaker", "unknown"),
                    "context": rec.get("context", "unknown"),
                    "original_label": raw_lbl,
                    "target_type": "fine_grained",
                })

        return statements, labels, metadata
