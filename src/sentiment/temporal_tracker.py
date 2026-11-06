"""Temporal Sentiment Tracker (Company Step 3).

Provides time-based aggregation of sentiment analysis results over configurable windows
(daily, weekly, monthly). Parses UNIX timestamps and ISO-8601 strings into normalized UTC,
computes sentiment distributions, percentages, and net sentiment scores.
"""

from collections import defaultdict
from datetime import datetime, timezone
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SentimentTemporalTracker:
    """Aggregates and tracks sentiment distributions over time windows."""

    @staticmethod
    def parse_timestamp(value: Any) -> Optional[datetime]:
        """Parses various timestamp representations into a UTC datetime object.

        Supports:
        - datetime objects (timezone-aware or naive assumed UTC)
        - UNIX epoch integers/floats (seconds)
        - Numeric strings (e.g. "1730101200")
        - ISO-8601 strings (e.g. "2026-11-01T12:00:00Z", "2026-11-05T16:54:13.868878+00:00")

        Args:
            value: Timestamp value to parse.

        Returns:
            UTC datetime object if successful, None otherwise.
        """
        if value is None:
            return None

        # Already a datetime
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        # Numeric (int or float timestamp)
        if isinstance(value, (int, float)):
            try:
                if math.isnan(value) or math.isinf(value):
                    return None
                # Support millisecond timestamps if timestamp > 1e11 (year ~5138 in seconds)
                ts_seconds = value / 1000.0 if value > 1e11 else value
                return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
            except (ValueError, OverflowError, OSError) as exc:
                logger.debug("Failed to parse numeric timestamp %s: %s", value, exc)
                return None

        # String representation
        if isinstance(value, str):
            val_str = value.strip()
            if not val_str:
                return None

            # Try numeric string (e.g. "1730101200")
            try:
                num = float(val_str)
                ts_seconds = num / 1000.0 if num > 1e11 else num
                return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                pass

            # Try ISO-8601 formats
            # Replace trailing 'Z' with UTC offset for fromisoformat compatibility
            clean_iso = val_str
            if clean_iso.endswith("Z") or clean_iso.endswith("z"):
                clean_iso = clean_iso[:-1] + "+00:00"

            try:
                dt = datetime.fromisoformat(clean_iso)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                pass

            # Fallback strptime patterns
            patterns = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
            ]
            for pat in patterns:
                try:
                    dt = datetime.strptime(val_str, pat)
                    if dt.tzinfo is None:
                        return dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc)
                except ValueError:
                    continue

        return None

    @staticmethod
    def format_window_key(dt: datetime, window: str = "daily") -> str:
        """Formats a datetime into a temporal bucket key.

        Args:
            dt: UTC datetime object.
            window: Time window type ('daily', 'weekly', 'monthly').

        Returns:
            String key representing the time bucket.
        """
        win = window.lower().strip()
        if win in ("weekly", "week"):
            iso_year, iso_week, _ = dt.isocalendar()
            return f"{iso_year}-W{iso_week:02d}"
        elif win in ("monthly", "month"):
            return dt.strftime("%Y-%m")
        else:
            # Default to daily
            return dt.strftime("%Y-%m-%d")

    @classmethod
    def calculate_window_metrics(cls, records: list[dict[str, Any]], sentiment_field: str = "sentiment_label") -> dict[str, Any]:
        """Calculates sentiment summary counts, percentages, and net score for a list of records.

        Args:
            records: List of record dictionaries.
            sentiment_field: Key containing the sentiment label.

        Returns:
            Dictionary with count and percentage breakdown.
        """
        total = len(records)
        pos = sum(1 for r in records if r.get(sentiment_field) == "positive")
        neu = sum(1 for r in records if r.get(sentiment_field) == "neutral")
        neg = sum(1 for r in records if r.get(sentiment_field) == "negative")
        und = total - (pos + neu + neg)

        pos_pct = round((pos / total) * 100.0, 2) if total > 0 else 0.0
        neu_pct = round((neu / total) * 100.0, 2) if total > 0 else 0.0
        neg_pct = round((neg / total) * 100.0, 2) if total > 0 else 0.0
        und_pct = round((und / total) * 100.0, 2) if total > 0 else 0.0

        # Net sentiment: (positive - negative) / total in range [-1.0, 1.0]
        net_sentiment = round((pos - neg) / total, 4) if total > 0 else 0.0

        return {
            "total": total,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "undetermined": und,
            "positive_pct": pos_pct,
            "neutral_pct": neu_pct,
            "negative_pct": neg_pct,
            "undetermined_pct": und_pct,
            "net_sentiment": net_sentiment,
        }

    @classmethod
    def aggregate_by_window(
        cls,
        records: list[dict[str, Any]],
        time_field: str = "created_utc",
        sentiment_field: str = "sentiment_label",
        window: str = "daily",
        fallback_time_field: Optional[str] = "collected_at",
    ) -> dict[str, Any]:
        """Groups records by time window and computes sentiment statistics per bucket.

        Args:
            records: List of record dictionaries.
            time_field: Primary field containing the timestamp.
            sentiment_field: Field containing the sentiment label.
            window: Aggregation window ('daily', 'weekly', 'monthly').
            fallback_time_field: Secondary timestamp field if primary is missing.

        Returns:
            Dictionary of time buckets sorted chronologically with sentiment metrics.
        """
        if not isinstance(records, list):
            raise TypeError("Expected records to be a list of dicts.")

        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            if not isinstance(record, dict):
                continue

            raw_time = record.get(time_field)
            if raw_time is None and fallback_time_field:
                raw_time = record.get(fallback_time_field)

            dt = cls.parse_timestamp(raw_time)
            if dt is None:
                bucket_key = "unknown"
            else:
                bucket_key = cls.format_window_key(dt, window=window)

            buckets[bucket_key].append(record)

        # Sort bucket keys: date keys sorted ascending, 'unknown' placed last
        regular_keys = sorted([k for k in buckets if k != "unknown"])
        ordered_keys = regular_keys + (["unknown"] if "unknown" in buckets else [])

        result: dict[str, Any] = {}
        for key in ordered_keys:
            result[key] = cls.calculate_window_metrics(buckets[key], sentiment_field=sentiment_field)

        return result
