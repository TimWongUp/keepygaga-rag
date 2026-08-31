from __future__ import annotations

from datetime import UTC, datetime, timedelta

from keepygaga_rag.knowledge.db import SourceRecord

PROVIDER_ACTION_RETRY_SECONDS = 60 * 60


def provider_action_retry_delay_seconds(error: str) -> int:
    if "ProviderActionRequiredError:" in error:
        return PROVIDER_ACTION_RETRY_SECONDS
    if any(
        legacy_status in error
        for legacy_status in (
            "401 Unauthorized",
            "402 Payment Required",
            "403 Forbidden",
        )
    ):
        return PROVIDER_ACTION_RETRY_SECONDS
    return 0


def automatic_retry_at(
    source: SourceRecord,
    *,
    regular_interval_seconds: int,
) -> datetime | None:
    if not source.last_scan_at:
        return None
    try:
        last_scan_at = datetime.fromisoformat(source.last_scan_at)
    except ValueError:
        return None
    if last_scan_at.tzinfo is None:
        last_scan_at = last_scan_at.replace(tzinfo=UTC)
    delay_seconds = max(
        regular_interval_seconds,
        provider_action_retry_delay_seconds(source.last_error),
    )
    return last_scan_at + timedelta(seconds=delay_seconds)


def automatic_retry_is_due(
    source: SourceRecord,
    *,
    regular_interval_seconds: int,
    now: datetime,
) -> bool:
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    retry_at = automatic_retry_at(
        source,
        regular_interval_seconds=regular_interval_seconds,
    )
    return retry_at is None or retry_at <= now
