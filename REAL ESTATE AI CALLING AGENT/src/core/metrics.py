"""Prometheus Metrics Exporter for Real Estate AI Voice Agent.

Exposes standard OpenMetrics/Prometheus exposition metrics at `/metrics`:
- Call volumes (started, completed, dropped)
- Active call sessions & active calendar holds
- End-to-end component latencies (VAD, STT, LLM, TTS)
- Operator notifications queue depth
- System error counters
"""

import time
from typing import Dict
from collections import defaultdict
from sqlalchemy import select, func
from src.db.session import AsyncSessionLocal
from src.db.models import CalendarSlot, SlotStatus, OperatorNotification


class MetricsRegistry:
    """In-memory metrics registry formatted for Prometheus scraping."""

    def __init__(self):
        self.calls_total = defaultdict(int)  # status -> count
        self.stage_latency_seconds = defaultdict(list)  # stage -> [latencies]
        self.errors_total = defaultdict(int)  # component -> count
        self.vad_turn_boundaries_total = 0
        self.confirmations_requested_total = 0

    def record_call_start(self):
        self.calls_total["started"] += 1

    def record_call_end(self):
        self.calls_total["completed"] += 1

    def record_call_drop(self):
        self.calls_total["dropped"] += 1

    def record_stage_latency(self, stage: str, seconds: float):
        # Keep rolling window of last 200 measurements
        latencies = self.stage_latency_seconds[stage]
        latencies.append(seconds)
        if len(latencies) > 200:
            latencies.pop(0)

    def record_error(self, component: str):
        self.errors_total[component] += 1

    def record_vad_boundary(self):
        self.vad_turn_boundaries_total += 1

    def record_confirmation_requested(self):
        self.confirmations_requested_total += 1

    async def generate_metrics_text(self) -> str:
        """Collects database gauges and formats OpenMetrics Prometheus output."""
        lines = []

        # 1. Calls total Counter
        lines.append("# HELP voice_agent_calls_total Total telephony calls handled by status.")
        lines.append("# TYPE voice_agent_calls_total counter")
        for status_label in ["started", "completed", "dropped"]:
            cnt = self.calls_total[status_label]
            lines.append(f'voice_agent_calls_total{{status="{status_label}"}} {cnt}')

        # 2. VAD Turn Boundaries Counter
        lines.append("# HELP voice_agent_vad_boundaries_total Total end-of-turn speech boundaries detected.")
        lines.append("# TYPE voice_agent_vad_boundaries_total counter")
        lines.append(f"voice_agent_vad_boundaries_total {self.vad_turn_boundaries_total}")

        # 3. Speech Uncertainty Confirmations Counter
        lines.append("# HELP voice_agent_confirmations_requested_total Total speech uncertainty safeguards triggered.")
        lines.append("# TYPE voice_agent_confirmations_requested_total counter")
        lines.append(f"voice_agent_confirmations_requested_total {self.confirmations_requested_total}")

        # 4. Errors total Counter
        lines.append("# HELP voice_agent_errors_total Total application errors by subsystem.")
        lines.append("# TYPE voice_agent_errors_total counter")
        if not self.errors_total:
            lines.append('voice_agent_errors_total{component="none"} 0')
        else:
            for comp, cnt in self.errors_total.items():
                lines.append(f'voice_agent_errors_total{{component="{comp}"}} {cnt}')

        # 5. Pipeline Latency Gauges and Prometheus Histograms
        lines.append("# HELP voice_agent_latency_seconds Average pipeline turnaround time in seconds.")
        lines.append("# TYPE voice_agent_latency_seconds gauge")
        for stage, vals in self.stage_latency_seconds.items():
            if vals:
                avg_val = sum(vals) / len(vals)
                p95_val = sorted(vals)[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0]
                lines.append(f'voice_agent_latency_seconds{{stage="{stage}",metric="avg"}} {avg_val:.4f}')
                lines.append(f'voice_agent_latency_seconds{{stage="{stage}",metric="p95"}} {p95_val:.4f}')

        lines.append("# HELP voice_agent_stage_duration_seconds Histogram of pipeline stage execution times in seconds.")
        lines.append("# TYPE voice_agent_stage_duration_seconds histogram")
        HISTOGRAM_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        for stage, vals in self.stage_latency_seconds.items():
            if vals:
                for b in HISTOGRAM_BUCKETS:
                    count_le = sum(1 for v in vals if v <= b)
                    lines.append(f'voice_agent_stage_duration_seconds_bucket{{stage="{stage}",le="{b}"}} {count_le}')
                lines.append(f'voice_agent_stage_duration_seconds_bucket{{stage="{stage}",le="+Inf"}} {len(vals)}')
                lines.append(f'voice_agent_stage_duration_seconds_sum{{stage="{stage}"}} {sum(vals):.4f}')
                lines.append(f'voice_agent_stage_duration_seconds_count{{stage="{stage}"}} {len(vals)}')

        # 6. Live Database Gauges
        active_holds = 0
        pending_notifications = 0
        confirmed_appointments = 0

        try:
            async with AsyncSessionLocal() as sess:
                # Active holds
                holds_query = select(func.count(CalendarSlot.id)).where(CalendarSlot.status == SlotStatus.HELD)
                active_holds = (await sess.execute(holds_query)).scalar() or 0

                # Confirmed visits
                confirmed_query = select(func.count(CalendarSlot.id)).where(CalendarSlot.status == SlotStatus.CONFIRMED)
                confirmed_appointments = (await sess.execute(confirmed_query)).scalar() or 0

                # Pending operator notifications
                notif_query = select(func.count(OperatorNotification.id)).where(OperatorNotification.status == "pending")
                pending_notifications = (await sess.execute(notif_query)).scalar() or 0
        except Exception:
            pass

        lines.append("# HELP voice_agent_active_holds Current active 5-minute reservation holds.")
        lines.append("# TYPE voice_agent_active_holds gauge")
        lines.append(f"voice_agent_active_holds {active_holds}")

        lines.append("# HELP voice_agent_confirmed_appointments_total Total confirmed property visit appointments.")
        lines.append("# TYPE voice_agent_confirmed_appointments_total gauge")
        lines.append(f"voice_agent_confirmed_appointments_total {confirmed_appointments}")

        lines.append("# HELP voice_agent_operator_notifications_pending Current pending human callback queue depth.")
        lines.append("# TYPE voice_agent_operator_notifications_pending gauge")
        lines.append(f"voice_agent_operator_notifications_pending {pending_notifications}")

        lines.append("")
        return "\n".join(lines)


# Global metrics registry singleton
metrics_registry = MetricsRegistry()
