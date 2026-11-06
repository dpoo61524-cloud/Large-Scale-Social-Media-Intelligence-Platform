"""Topic Temporal Tracker (Company Step 4).

Tracks topic popularity and discussion volume over time windows (daily/weekly/monthly).
Computes advanced temporal dynamics: topic velocity, acceleration, rolling moving average,
breakout status classification, and cross-platform divergence (Reddit vs. YouTube) via
Jensen-Shannon divergence and adoption gaps.
"""

from collections import defaultdict
from datetime import datetime, timezone
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

STATUS_BREAKOUT = "breakout"
STATUS_SUSTAINED_RISE = "sustained_rise"
STATUS_PLATEAU = "plateau"
STATUS_DECAY = "decay"
STATUS_DORMANT = "dormant"
STATUS_EMERGING = "emerging"

TRAJECTORY_RISE = "rise"
TRAJECTORY_FALL = "fall"
TRAJECTORY_STABLE = "stable"
TRAJECTORY_EMERGING = "emerging"


class TopicTemporalTracker:
    """Advanced tracker for topic temporal dynamics, acceleration, and cross-platform divergence."""

    @staticmethod
    def parse_timestamp(value: Any) -> Optional[datetime]:
        """Parses various timestamp representations into a UTC datetime object.

        Supports:
        - datetime instances
        - UNIX epoch integers/floats (seconds or milliseconds)
        - Numeric strings (e.g. "1730101200")
        - ISO-8601 strings (e.g. "2026-11-01T12:00:00Z")

        Args:
            value: Timestamp value to parse.

        Returns:
            UTC datetime object if successfully parsed, None otherwise.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, (int, float)):
            try:
                if math.isnan(value) or math.isinf(value):
                    return None
                ts_seconds = value / 1000.0 if value > 1e11 else value
                return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
            except (ValueError, OverflowError, OSError) as exc:
                logger.debug("Failed to parse numeric timestamp %s: %s", value, exc)
                return None

        if isinstance(value, str):
            val_str = value.strip()
            if not val_str:
                return None

            try:
                num = float(val_str)
                ts_seconds = num / 1000.0 if num > 1e11 else num
                return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                pass

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

            patterns = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d",
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
            String representing the temporal bucket.
        """
        win = window.lower().strip()
        if win in ("weekly", "week"):
            iso_year, iso_week, _ = dt.isocalendar()
            return f"{iso_year}-W{iso_week:02d}"
        elif win in ("monthly", "month"):
            return dt.strftime("%Y-%m")
        else:
            return dt.strftime("%Y-%m-%d")

    @classmethod
    def track_topic_frequency_over_time(
        cls,
        records: list[dict[str, Any]],
        topic_field: str = "topic_id",
        time_field: str = "created_utc",
        window: str = "daily",
        fallback_time_field: Optional[str] = "collected_at",
    ) -> dict[str, Any]:
        """Groups records by temporal window and calculates topic frequencies and shares.

        Args:
            records: List of processed document records.
            topic_field: Field name containing assigned topic ID.
            time_field: Field name containing the primary timestamp.
            window: Temporal aggregation window ('daily', 'weekly', 'monthly').
            fallback_time_field: Fallback timestamp field if primary is missing.

        Returns:
            Dictionary containing window breakdowns and topic time-series.
        """
        if not isinstance(records, list):
            raise TypeError("Expected records to be a list of dicts.")

        if not records:
            return {
                "windows": {},
                "topic_summaries": {},
                "all_topics": [],
                "time_windows": [],
            }

        window_topic_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        window_totals: dict[str, int] = defaultdict(int)
        all_topics_set: set[str] = set()

        for rec in records:
            if not isinstance(rec, dict):
                continue

            raw_topic = rec.get(topic_field, -1)
            topic_key = str(raw_topic)
            all_topics_set.add(topic_key)

            raw_time = rec.get(time_field)
            if raw_time is None and fallback_time_field:
                raw_time = rec.get(fallback_time_field)

            dt = cls.parse_timestamp(raw_time)
            bucket_key = "unknown" if dt is None else cls.format_window_key(dt, window=window)

            window_topic_counts[bucket_key][topic_key] += 1
            window_totals[bucket_key] += 1

        date_keys = sorted([k for k in window_totals if k != "unknown"])
        ordered_windows = date_keys + (["unknown"] if "unknown" in window_totals else [])

        windows_output: dict[str, Any] = {}
        for win_key in ordered_windows:
            total_docs = window_totals[win_key]
            t_counts = dict(window_topic_counts[win_key])
            t_shares = {
                tid: round((cnt / total_docs) * 100.0, 2) if total_docs > 0 else 0.0
                for tid, cnt in t_counts.items()
            }
            windows_output[win_key] = {
                "total_documents": total_docs,
                "topic_counts": t_counts,
                "topic_shares": t_shares,
            }

        sorted_topics = sorted(all_topics_set, key=lambda x: (int(x) if x.lstrip("-").isdigit() else 9999))
        topic_summaries: dict[str, Any] = {}

        for tid in sorted_topics:
            counts_per_window: dict[str, int] = {}
            total_count = 0
            for win_key in ordered_windows:
                cnt = window_topic_counts[win_key].get(tid, 0)
                counts_per_window[win_key] = cnt
                total_count += cnt

            topic_summaries[tid] = {
                "total_count": total_count,
                "window_counts": counts_per_window,
            }

        return {
            "windows": windows_output,
            "topic_summaries": topic_summaries,
            "all_topics": sorted_topics,
            "time_windows": ordered_windows,
        }

    @classmethod
    def calculate_topic_dynamics(
        cls,
        temporal_data: dict[str, Any],
        growth_threshold_pct: float = 10.0,
        smoothing_window: int = 3,
    ) -> dict[str, Any]:
        """Calculates advanced temporal dynamics: velocity, acceleration, and momentum status.

        Args:
            temporal_data: Output dictionary from track_topic_frequency_over_time.
            growth_threshold_pct: Minimum percentage growth for rise categorization.
            smoothing_window: Window size for rolling moving-average smoothing.

        Returns:
            Dictionary mapping topic_id to detailed velocity, acceleration, and status dynamics.
        """
        windows = [w for w in temporal_data.get("time_windows", []) if w != "unknown"]
        topic_summaries = temporal_data.get("topic_summaries", {})
        results: dict[str, Any] = {}

        for tid, summary in topic_summaries.items():
            win_counts = summary.get("window_counts", {})
            series = [win_counts.get(w, 0) for w in windows]

            # Calculate rolling moving average
            moving_averages: list[float] = []
            for i in range(len(series)):
                start_idx = max(0, i - smoothing_window + 1)
                sub_slice = series[start_idx : i + 1]
                avg_val = sum(sub_slice) / len(sub_slice) if sub_slice else 0.0
                moving_averages.append(round(avg_val, 2))

            latest_ma = moving_averages[-1] if moving_averages else 0.0

            if len(windows) < 2:
                latest_cnt = series[0] if series else 0
                results[tid] = {
                    "status": STATUS_EMERGING if latest_cnt > 0 else STATUS_DORMANT,
                    "velocity": 0,
                    "acceleration": 0,
                    "growth_rate_pct": 0.0,
                    "moving_average": latest_ma,
                    "latest_count": latest_cnt,
                    "previous_count": 0,
                    "latest_window": windows[0] if windows else "none",
                    "previous_window": None,
                }
                continue

            curr_cnt = series[-1]
            prev_cnt = series[-2]
            v_curr = curr_cnt - prev_cnt

            # Compute acceleration if at least 3 windows exist
            if len(series) >= 3:
                prev_prev_cnt = series[-3]
                v_prev = prev_cnt - prev_prev_cnt
                acceleration = v_curr - v_prev
            else:
                acceleration = v_curr

            # Growth rate
            if prev_cnt > 0:
                growth_rate = round(((curr_cnt - prev_cnt) / prev_cnt) * 100.0, 2)
            else:
                growth_rate = 100.0 if curr_cnt > 0 else 0.0

            # Momentum status categorization
            if curr_cnt == 0 and prev_cnt == 0:
                status = STATUS_DORMANT
            elif prev_cnt == 0 and curr_cnt > 0:
                status = STATUS_EMERGING
            elif v_curr > 0 and acceleration > 0:
                status = STATUS_BREAKOUT
            elif v_curr > 0 and acceleration <= 0:
                status = STATUS_SUSTAINED_RISE
            elif v_curr == 0:
                status = STATUS_PLATEAU
            else:
                status = STATUS_DECAY

            results[tid] = {
                "status": status,
                "velocity": v_curr,
                "acceleration": acceleration,
                "growth_rate_pct": growth_rate,
                "moving_average": latest_ma,
                "latest_count": curr_cnt,
                "previous_count": prev_cnt,
                "latest_window": windows[-1],
                "previous_window": windows[-2],
            }

        return results

    @classmethod
    def calculate_topic_trajectories(
        cls,
        temporal_data: dict[str, Any],
        growth_threshold_pct: float = 10.0,
    ) -> dict[str, Any]:
        """Backward-compatible trajectory wrapper integrating with advanced dynamics."""
        dynamics = cls.calculate_topic_dynamics(temporal_data, growth_threshold_pct=growth_threshold_pct)
        trajectories: dict[str, Any] = {}

        for tid, dyn in dynamics.items():
            status = dyn["status"]
            velocity = dyn["velocity"]
            growth_rate = dyn["growth_rate_pct"]

            if status == STATUS_BREAKOUT or status == STATUS_SUSTAINED_RISE or (velocity > 0 and growth_rate >= growth_threshold_pct):
                traj = TRAJECTORY_RISE
            elif status == STATUS_DECAY or (velocity < 0 and growth_rate <= -growth_threshold_pct):
                traj = TRAJECTORY_FALL
            elif status == STATUS_EMERGING:
                traj = TRAJECTORY_EMERGING
            else:
                traj = TRAJECTORY_STABLE

            trajectories[tid] = {
                "trajectory": traj,
                "velocity": velocity,
                "acceleration": dyn["acceleration"],
                "status": status,
                "growth_rate_pct": growth_rate,
                "latest_count": dyn["latest_count"],
                "previous_count": dyn["previous_count"],
                "latest_window": dyn["latest_window"],
                "previous_window": dyn["previous_window"],
            }

        return trajectories

    @classmethod
    def calculate_cross_platform_divergence(
        cls,
        records: list[dict[str, Any]],
        platform_field: str = "source_platform",
        topic_field: str = "topic_id",
        time_field: str = "created_utc",
        window: str = "daily",
    ) -> dict[str, Any]:
        """Calculates topic adoption divergence and Jensen-Shannon divergence between platforms.

        Args:
            records: List of documents tagged with platform ('reddit' vs 'youtube').
            platform_field: Key indicating source platform.
            topic_field: Key indicating topic assignment.
            time_field: Primary timestamp field.
            window: Temporal aggregation window.

        Returns:
            Dictionary containing overall JS divergence, adoption gap per topic, and temporal divergence.
        """
        if not isinstance(records, list) or not records:
            return {
                "overall_divergence": 0.0,
                "topic_adoption_gaps": {},
                "window_divergences": {},
            }

        reddit_records = [r for r in records if r.get(platform_field) == "reddit"]
        youtube_records = [r for r in records if r.get(platform_field) == "youtube"]

        all_topic_ids = sorted(list({str(r.get(topic_field, -1)) for r in records}))

        # 1. Overall platform topic distributions
        total_reddit = len(reddit_records)
        total_youtube = len(youtube_records)

        reddit_counts = defaultdict(int)
        for r in reddit_records:
            reddit_counts[str(r.get(topic_field, -1))] += 1

        youtube_counts = defaultdict(int)
        for r in youtube_records:
            youtube_counts[str(r.get(topic_field, -1))] += 1

        p_reddit = [reddit_counts[tid] / total_reddit if total_reddit > 0 else 0.0 for tid in all_topic_ids]
        q_youtube = [youtube_counts[tid] / total_youtube if total_youtube > 0 else 0.0 for tid in all_topic_ids]

        overall_js_div = cls._compute_jensen_shannon_divergence(p_reddit, q_youtube)

        # 2. Topic adoption gaps
        adoption_gaps: dict[str, Any] = {}
        for idx, tid in enumerate(all_topic_ids):
            r_share = round(p_reddit[idx] * 100.0, 2)
            y_share = round(q_youtube[idx] * 100.0, 2)
            gap = round(r_share - y_share, 2)

            if gap > 5.0:
                over_indexed = "reddit"
            elif gap < -5.0:
                over_indexed = "youtube"
            else:
                over_indexed = "balanced"

            adoption_gaps[tid] = {
                "reddit_share_pct": r_share,
                "youtube_share_pct": y_share,
                "gap_pct": gap,
                "over_indexed_on": over_indexed,
            }

        # 3. Temporal window divergence
        reddit_temporal = cls.track_topic_frequency_over_time(reddit_records, topic_field=topic_field, time_field=time_field, window=window)
        youtube_temporal = cls.track_topic_frequency_over_time(youtube_records, topic_field=topic_field, time_field=time_field, window=window)

        all_windows = sorted(list(set(reddit_temporal["time_windows"] + youtube_temporal["time_windows"])))
        window_divergences: dict[str, Any] = {}

        for win in all_windows:
            r_win_counts = reddit_temporal["windows"].get(win, {}).get("topic_counts", {})
            y_win_counts = youtube_temporal["windows"].get(win, {}).get("topic_counts", {})

            r_tot = sum(r_win_counts.values())
            y_tot = sum(y_win_counts.values())

            if r_tot > 0 and y_tot > 0:
                p_win = [r_win_counts.get(tid, 0) / r_tot for tid in all_topic_ids]
                q_win = [y_win_counts.get(tid, 0) / y_tot for tid in all_topic_ids]
                js_win = cls._compute_jensen_shannon_divergence(p_win, q_win)
            else:
                js_win = 0.0

            window_divergences[win] = {
                "reddit_total": r_tot,
                "youtube_total": y_tot,
                "jensen_shannon_divergence": round(js_win, 4),
            }

        return {
            "overall_divergence": round(overall_js_div, 4),
            "topic_adoption_gaps": adoption_gaps,
            "window_divergences": window_divergences,
        }

    @staticmethod
    def _compute_jensen_shannon_divergence(p: list[float], q: list[float]) -> float:
        """Computes Jensen-Shannon divergence between probability distributions P and Q."""
        if not p or not q or len(p) != len(q):
            return 0.0

        eps = 1e-12
        m = [(p[i] + q[i]) / 2.0 for i in range(len(p))]

        def kl_divergence(dist: list[float], ref: list[float]) -> float:
            kl = 0.0
            for d, r in zip(dist, ref):
                if d > eps and r > eps:
                    # Log base 2 so JS divergence is strictly in [0.0, 1.0]
                    kl += d * math.log2(d / r)
            return kl

        jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        return max(0.0, min(1.0, jsd))