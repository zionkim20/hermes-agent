"""Automatic Kanban capture for actionable gateway chat intake.

This module is intentionally small and synchronous: it runs in the gateway
before the expensive agent turn so household work is durably captured even if
the model run, context compression, or process restart fails later.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = __import__("logging").getLogger(__name__)

_ACTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(please|pls)\s+(book|schedule|order|buy|get|find|check|call|text|send|remind|make|create|add|update|cancel|reschedule|plan|prepare|handle|follow\s*up|look\s+into|monitor|compare|save|protect)\b",
        r"^(book|schedule|order|buy|get|find|check|call|text|send|remind|make|create|add|update|cancel|reschedule|plan|prepare|handle|follow\s*up|look\s+into|monitor|compare|save|protect)\b",
        r"\b(can you|could you|would you|will you)\s+(book|schedule|order|buy|get|find|check|call|text|send|remind|make|create|add|update|cancel|reschedule|plan|prepare|handle|follow\s*up|look\s+into|monitor|compare|save|protect)\b",
        r"\b(need|needs)\s+(to\s+be\s+)?(booked|scheduled|ordered|bought|checked|handled|done|updated|cancelled|canceled|rescheduled)\b",
        r"\b(remind me|reminder to|todo:|to do:)\b",
    )
)

_NON_ACTIONABLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(thanks?|thank you|ty|ok|okay|yes|no|got it|perfect|great|done)[.!\s]*$",
        r"^(hi|hello|hey|bom dia|boa tarde|boa noite)[.!\s]*$",
    )
)

_LEADING_POLITENESS_RE = re.compile(
    r"^(?:\[[^\]]+\]\s*)?(?:(?:please|pls)\s+)?(?:(?:can|could|would|will)\s+you\s+)?",
    re.IGNORECASE,
)

INVISIBLE_LOAD_TYPES: tuple[str, ...] = (
    "identifying",
    "noticing",
    "anticipation",
    "monitoring",
    "remembering",
    "deciding",
    "researching",
    "coordinating",
    "following_up",
    "emotional_containment",
    "identity_values_work",
    "crisis_prepositioning",
    "health_surveillance",
    "financial_cognitive_load",
    "memory_keepsake",
    "carrier_self_care",
)

MENTAL_LOAD_CATEGORIES: tuple[str, ...] = (
    "execution",
    "anticipation",
    "decision_research",
    "relationship_maintenance",
    "emotional_labor",
    "identity_values_work",
    "crisis_prepositioning",
    "health_surveillance",
    "financial_cognitive_load",
    "memory_keepsake_carrier_self_care",
)


@dataclass(frozen=True)
class GatewayKanbanIntakeConfig:
    """Configuration for writing inbound gateway tasks to Kanban."""

    enabled: bool = False
    assignee: str = "miarouter"
    tenant: Optional[str] = None
    created_by: str = "gateway-intake"
    board: Optional[str] = None
    priority: int = 50
    max_title_chars: int = 90
    allowed_user_ids: tuple[str, ...] = ()
    allowed_user_names: tuple[str, ...] = ()
    allowed_chat_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Optional[dict[str, Any]]) -> "GatewayKanbanIntakeConfig":
        if not isinstance(data, dict):
            return cls()

        def _tuple(key: str) -> tuple[str, ...]:
            value = data.get(key) or ()
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple, set)):
                return ()
            return tuple(str(item).strip() for item in value if str(item).strip())

        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            assignee=str(data.get("assignee") or "miarouter"),
            tenant=_optional_str(data.get("tenant")),
            created_by=str(data.get("created_by") or "gateway-intake"),
            board=_optional_str(data.get("board")),
            priority=_coerce_int(data.get("priority"), 50),
            max_title_chars=max(20, _coerce_int(data.get("max_title_chars"), 90)),
            allowed_user_ids=_tuple("allowed_user_ids"),
            allowed_user_names=_tuple("allowed_user_names"),
            allowed_chat_ids=_tuple("allowed_chat_ids"),
        )

    @classmethod
    def from_gateway_config(cls, gateway_config: Any) -> "GatewayKanbanIntakeConfig":
        """Read kanban_intake config from a GatewayConfig or raw dict.

        Supported YAML shapes:
          kanban_intake: {enabled: true, ...}
          gateway: {kanban_intake: {enabled: true, ...}}
        """
        if isinstance(gateway_config, dict):
            block = gateway_config.get("kanban_intake")
            if block is None and isinstance(gateway_config.get("gateway"), dict):
                block = gateway_config["gateway"].get("kanban_intake")
            return cls.from_mapping(block)
        return getattr(gateway_config, "kanban_intake", cls())


@dataclass(frozen=True)
class KanbanIntakeCaptureResult:
    created: bool
    task_id: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class CapabilityBlockerCaptureResult:
    """Result of post-turn missing-capability capture."""

    captured: bool
    task_id: Optional[str] = None
    created: bool = False
    capability: str = ""
    use_case: str = ""
    title: str = ""
    reason: str = ""


@dataclass(frozen=True)
class CapabilityBlocker:
    capability: str
    use_case: str
    current_blocker: str
    smallest_unblock_step: str
    acceptance_test: str
    trigger_phrase: str


@dataclass(frozen=True)
class ActiveKanbanSweepResult:
    """Best-effort board sweep performed while an authorized user is present."""

    scanned: bool = False
    reason: str = ""
    unblocked_task_ids: tuple[str, ...] = ()
    human_blocked_task_ids: tuple[str, ...] = ()
    credential_blocked_task_ids: tuple[str, ...] = ()
    external_blocked_task_ids: tuple[str, ...] = ()
    dispatch_spawned_task_ids: tuple[str, ...] = ()
    dispatch_reclaimed_task_ids: tuple[str, ...] = ()
    context_note: str = ""


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def should_capture_actionable_task(text: str) -> bool:
    """Heuristic guard for actionable household requests.

    This deliberately favors recall over precision once the feature is enabled,
    but filters obvious acknowledgements and slash commands so the queue does not
    fill with conversational glue.
    """
    normalized = " ".join((text or "").strip().split())
    if not normalized or normalized.startswith("/"):
        return False
    if len(normalized) < 4:
        return False
    if any(pattern.search(normalized) for pattern in _NON_ACTIONABLE_PATTERNS):
        return False
    return any(pattern.search(normalized) for pattern in _ACTION_PATTERNS)


def should_scan_active_kanban(text: str) -> bool:
    """Return True for meaningful user presence that can unblock stale work."""
    normalized = " ".join((text or "").strip().split())
    if not normalized or normalized.startswith("/"):
        return False
    if len(normalized) < 4:
        return False
    if any(pattern.search(normalized) for pattern in _NON_ACTIONABLE_PATTERNS):
        return False
    return True


def active_interaction_kanban_sweep(
    event: Any,
    config: GatewayKanbanIntakeConfig,
    *,
    prepared_text: Optional[str] = None,
    spawn_fn: Any = None,
    max_spawn: int = 1,
) -> ActiveKanbanSweepResult:
    """Sweep household Kanban when Zion/Renata are actively present."""
    if not config.enabled:
        return ActiveKanbanSweepResult(reason="disabled")
    if bool(getattr(event, "internal", False)):
        return ActiveKanbanSweepResult(reason="internal")

    source = getattr(event, "source", None)
    if not _source_allowed(source, config):
        return ActiveKanbanSweepResult(reason="source_not_allowed")

    text = prepared_text if prepared_text is not None else getattr(event, "text", "")
    if not should_scan_active_kanban(text):
        return ActiveKanbanSweepResult(reason="not_meaningful")

    from hermes_cli import kanban_db

    conn = kanban_db.connect(board=config.board)
    try:
        unblocked: list[str] = []
        human: list[Any] = []
        credential: list[str] = []
        external: list[str] = []

        blocked_tasks = kanban_db.list_tasks(
            conn,
            status="blocked",
            tenant=config.tenant,
            limit=25,
            order_by="priority",
        )
        for task in blocked_tasks:
            reason_text = _blocked_task_reason(conn, task)
            blocker = _classify_blocker(task, reason_text)
            if blocker in {"worker", "not_blocked"}:
                if kanban_db.unblock_task(conn, task.id):
                    kanban_db.add_comment(
                        conn,
                        task.id,
                        "kanban-unblock-sweep",
                        "Active interaction sweep unblocked this task because the blocker was not Zion/Renata.",
                    )
                    unblocked.append(task.id)
                continue
            if blocker == "human":
                human.append((task, reason_text))
            elif blocker == "credential":
                credential.append(task.id)
            else:
                external.append(task.id)

        dispatch_result = kanban_db.dispatch_once(
            conn,
            spawn_fn=spawn_fn,
            max_spawn=max(0, int(max_spawn)),
            board=config.board,
            tenant=config.tenant,
        )

        spawned = tuple(task_id for task_id, _assignee, _workspace in dispatch_result.spawned)
        reclaimed = tuple(
            list(getattr(dispatch_result, "reclaimed", []) or [])
            + list(getattr(dispatch_result, "stale", []) or [])
            + list(getattr(dispatch_result, "crashed", []) or [])
            + list(getattr(dispatch_result, "timed_out", []) or [])
        )
        note = _build_active_sweep_context_note(
            unblocked=unblocked,
            human=human,
            spawned=list(spawned),
            reclaimed=list(reclaimed),
        )
        return ActiveKanbanSweepResult(
            scanned=True,
            reason="scanned",
            unblocked_task_ids=tuple(unblocked),
            human_blocked_task_ids=tuple(task.id for task, _reason in human),
            credential_blocked_task_ids=tuple(credential),
            external_blocked_task_ids=tuple(external),
            dispatch_spawned_task_ids=spawned,
            dispatch_reclaimed_task_ids=reclaimed,
            context_note=note,
        )
    finally:
        conn.close()


def _blocked_task_reason(conn: Any, task: Any) -> str:
    parts = [task.title or "", task.body or "", task.result or "", task.last_failure_error or ""]
    try:
        row = conn.execute(
            "SELECT summary, error FROM task_runs WHERE task_id = ? AND outcome = 'blocked' ORDER BY ended_at DESC, id DESC LIMIT 1",
            (task.id,),
        ).fetchone()
        if row:
            parts.extend([row["summary"] or "", row["error"] or ""])
    except Exception:
        pass
    try:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY created_at DESC, id DESC LIMIT 1",
            (task.id,),
        ).fetchone()
        if row and row["payload"]:
            parts.append(str(row["payload"]))
    except Exception:
        pass
    return "\n".join(part for part in parts if part)


def _classify_blocker(task: Any, reason_text: str) -> str:
    text = f"{task.title or ''}\n{reason_text or ''}".casefold()
    human_terms = (
        "zion", "renata", "human", "user", "approval", "approve", "confirm",
        "ratify", "review-required", "needs eyes", "spend", "cost",
        "pay", "purchase", "buy", "order", "book", "booking", "send email",
        "external", "school", "zaya", "kid", "child", "medical", "doctor",
        "irreversible", "relationship",
    )
    if any(term in text for term in human_terms):
        return "human"
    credential_terms = (
        "credential", "password", "login", "auth", "token", "api key",
        "not connected", "connect", "oauth", "permission", "access denied",
    )
    if any(term in text for term in credential_terms):
        return "credential"
    external_terms = (
        "waiting for vendor", "waiting on vendor", "third party", "reply from",
        "response from", "waiting for reply", "paywall", "document unavailable",
    )
    if any(term in text for term in external_terms):
        return "external"
    worker_terms = (
        "worker", "tool failure", "crash", "crashed", "timed out", "timeout",
        "spawn_failed", "spawn failure", "gateway shutdown", "context compaction",
        "interrupted", "safe to retry", "transient", "quota", "rate limit",
        "not actually blocked", "can continue",
    )
    if any(term in text for term in worker_terms):
        return "worker"
    return "external"


_CAPABILITY_BLOCKER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(blocked|could not complete|couldn['’]?t complete|cannot complete|can['’]?t complete)\b",
        r"\b(no access|access denied|missing access|permission denied)\b",
        r"\b(not configured|isn['’]?t configured|not set up|isn['’]?t set up)\b",
        r"\b(need(?:ed)? credential|missing credential|need(?:ed)? api key|missing api key|invalid[-\s]?key|invalid key|connector invalid|invalid connector|oauth.*invalid)\b",
        r"\b(missing capability|capability (?:is )?missing|tool (?:is )?missing|integration (?:is )?missing)\b",
    )
)


_CAPABILITY_BLOCKER_FALSE_POSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        # Status/cleanup language that mentions the word "Blocked" as a label
        # that was removed, not as a current inability to act. Without this,
        # routine success replies like "No more Needs you / Blocked confusion"
        # create fake connector-unblock cards.
        r"\bno more\b[^.!?\n]{0,160}\bblocked\b[^.!?\n]{0,80}\b(confusion|section|split|label|wording)\b",
        r"\bblocked\b[^.!?\n]{0,80}\b(confusion|section|split|label|wording)\b",
    )
)


def capture_capability_blocker_to_kanban(
    event: Any,
    config: GatewayKanbanIntakeConfig,
    *,
    final_response: str,
    attempted_text: Optional[str] = None,
    session_id: Optional[str] = None,
    session_key: Optional[str] = None,
    parent_task_id: Optional[str] = None,
) -> CapabilityBlockerCaptureResult:
    """Create or update a durable blocker when a turn reveals missing capability.

    This runs after the agent has produced its final reply but before the
    gateway delivers that reply. It catches the class of misses where the model
    tells the user "I couldn't do this" and would otherwise leave the blocker
    only in chat history.
    """
    if not config.enabled:
        return CapabilityBlockerCaptureResult(False, reason="disabled")
    if bool(getattr(event, "internal", False)):
        return CapabilityBlockerCaptureResult(False, reason="internal")

    source = getattr(event, "source", None)
    if not _source_allowed(source, config):
        return CapabilityBlockerCaptureResult(False, reason="source_not_allowed")

    blocker = detect_capability_blocker(final_response, attempted_text=attempted_text)
    if blocker is None:
        return CapabilityBlockerCaptureResult(False, reason="no_blocker_detected")

    capability_key = _slugify_blocker_key(blocker.capability)
    use_case_key = _slugify_blocker_key(blocker.use_case)
    idempotency_key = f"capability-blocker:{capability_key}:{use_case_key}"
    title = f"Unblock {blocker.capability}: {blocker.use_case}"

    from hermes_cli import kanban_db

    conn = kanban_db.connect(board=config.board)
    try:
        existing = _existing_task_id(conn, idempotency_key)
        if existing:
            kanban_db.add_comment(
                conn,
                existing,
                "capability-blocker-capture",
                _capability_blocker_update_comment(
                    event,
                    source,
                    blocker,
                    final_response=final_response,
                    attempted_text=attempted_text,
                    session_id=session_id,
                    session_key=session_key,
                    parent_task_id=parent_task_id,
                ),
            )
            return CapabilityBlockerCaptureResult(
                True,
                task_id=existing,
                created=False,
                capability=blocker.capability,
                use_case=blocker.use_case,
                title=title,
                reason="updated",
            )

        parents = (parent_task_id,) if parent_task_id else ()
        task_id = kanban_db.create_task(
            conn,
            title=title,
            body=_capability_blocker_body(
                event,
                source,
                blocker,
                final_response=final_response,
                attempted_text=attempted_text,
                session_id=session_id,
                session_key=session_key,
                parent_task_id=parent_task_id,
            ),
            assignee=config.assignee,
            created_by="capability-blocker-capture",
            tenant=config.tenant,
            priority=max(config.priority, 90),
            parents=parents,
            idempotency_key=idempotency_key,
            initial_status="blocked",
            session_id=session_id,
            board=config.board,
        )
        return CapabilityBlockerCaptureResult(
            True,
            task_id=task_id,
            created=True,
            capability=blocker.capability,
            use_case=blocker.use_case,
            title=title,
            reason="created",
        )
    finally:
        conn.close()


def detect_capability_blocker(
    final_response: str,
    *,
    attempted_text: Optional[str] = None,
) -> Optional[CapabilityBlocker]:
    """Best-effort missing-capability detector for final user replies."""
    response = " ".join((final_response or "").split())
    if not response:
        return None
    matched = next((pattern for pattern in _CAPABILITY_BLOCKER_PATTERNS if pattern.search(response)), None)
    if matched is None:
        return None
    if any(pattern.search(response) for pattern in _CAPABILITY_BLOCKER_FALSE_POSITIVE_PATTERNS):
        return None

    combined = f"{attempted_text or ''}\n{response}".casefold()
    if "vapi" in combined or "outbound call" in combined or "phone call" in combined or "telephony" in combined:
        return CapabilityBlocker(
            capability="Vapi outbound calls",
            use_case="live phone-call execution",
            current_blocker="Outbound-call/telephony capability is missing, unavailable, or not configured for this runtime.",
            smallest_unblock_step="Configure and verify the outbound calling connector, then tell Mia the capability is ready to test.",
            acceptance_test="From the gateway, place a real test outbound call through Vapi and confirm the call reaches the target number with a recorded success status.",
            trigger_phrase=matched.pattern,
        )
    if "composio" in combined and ("calendar" in combined or "heartbeat" in combined or "google" in combined):
        return CapabilityBlocker(
            capability="Composio calendar access",
            use_case="calendar heartbeat and calendar reads",
            current_blocker="Composio calendar access failed because the connector/API key is invalid, expired, or not connected.",
            smallest_unblock_step="Refresh or reconnect the Composio Google Calendar connector for the household account, then run a calendar heartbeat check.",
            acceptance_test="Run the calendar heartbeat against Zion and Renata calendar connections and verify both return current calendar data without invalid-key errors.",
            trigger_phrase=matched.pattern,
        )

    capability = _infer_generic_capability(combined)
    use_case = _infer_generic_use_case(attempted_text or response)
    return CapabilityBlocker(
        capability=capability,
        use_case=use_case,
        current_blocker="The attempted task surfaced a missing access, credential, connector, configuration, or runtime capability.",
        smallest_unblock_step="Identify the missing connector/credential/configuration, enable it, then rerun the exact user task as a real verification.",
        acceptance_test="Repeat the original task from the gateway and verify it completes without a blocked/no-access/not-configured reply.",
        trigger_phrase=matched.pattern,
    )


def _infer_generic_capability(text: str) -> str:
    known = (
        ("whatsapp", "WhatsApp connector"),
        ("telegram", "Telegram connector"),
        ("gmail", "Gmail connector"),
        ("email", "email connector"),
        ("drive", "Google Drive connector"),
        ("sheets", "Google Sheets connector"),
        ("calendar", "calendar connector"),
        ("1password", "1Password access"),
        ("op ", "1Password access"),
        ("browser", "browser automation"),
        ("payment", "payment capability"),
        ("credential", "credential access"),
        ("api key", "API key access"),
    )
    for needle, label in known:
        if needle in text:
            return label
    return "missing runtime capability"


def _infer_generic_use_case(text: str) -> str:
    cleaned = " ".join((text or "").split())
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned).strip()
    if not cleaned:
        return "original user task"
    return cleaned[:80].rstrip(" .,;:!?") or "original user task"


def _slugify_blocker_key(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").casefold()).strip("-")
    return slug[:80] or "unknown"


def _capability_blocker_body(
    event: Any,
    source: Any,
    blocker: CapabilityBlocker,
    *,
    final_response: str,
    attempted_text: Optional[str],
    session_id: Optional[str],
    session_key: Optional[str],
    parent_task_id: Optional[str],
) -> str:
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", "unknown"))
    lines = [
        "Automatic Kanban capture from a blocked execution reply.",
        "",
        "Capability/use-case:",
        f"capability: {blocker.capability}",
        f"use_case: {blocker.use_case}",
        "",
        "Source:",
        f"source_platform: {platform}",
        f"source_chat_id: {getattr(source, 'chat_id', '') or ''}",
        f"source_thread_id: {getattr(source, 'thread_id', '') or ''}",
        f"source_message_id: {getattr(event, 'message_id', None) or getattr(source, 'message_id', '') or ''}",
        f"sender_id: {getattr(source, 'user_id', '') or ''}",
        f"sender_name: {getattr(source, 'user_name', '') or ''}",
        f"session_id: {session_id or ''}",
        f"session_key: {session_key or ''}",
        f"parent_household_task: {parent_task_id or 'unknown'}",
        "",
        "Current blocker:",
        blocker.current_blocker,
        "",
        "Smallest human unblock step:",
        blocker.smallest_unblock_step,
        "",
        "Acceptance test for real verification:",
        blocker.acceptance_test,
        "",
        "Attempted task:",
        (attempted_text or "").strip() or "unknown",
        "",
        "Blocked reply excerpt:",
        final_response.strip()[:1200],
    ]
    return "\n".join(lines).strip() + "\n"


def _capability_blocker_update_comment(
    event: Any,
    source: Any,
    blocker: CapabilityBlocker,
    *,
    final_response: str,
    attempted_text: Optional[str],
    session_id: Optional[str],
    session_key: Optional[str],
    parent_task_id: Optional[str],
) -> str:
    return _capability_blocker_body(
        event,
        source,
        blocker,
        final_response=final_response,
        attempted_text=attempted_text,
        session_id=session_id,
        session_key=session_key,
        parent_task_id=parent_task_id,
    ).replace(
        "Automatic Kanban capture from a blocked execution reply.",
        "Repeated blocked execution mention; updating existing capability blocker.",
        1,
    )


def _build_active_sweep_context_note(
    *,
    unblocked: list[str],
    human: list[Any],
    spawned: list[str],
    reclaimed: list[str],
) -> str:
    if not (unblocked or human or spawned or reclaimed):
        return ""
    lines = [
        "[Active Kanban sweep]",
        "Meaningful Zion/Renata interaction detected; I inspected the household Kanban board.",
    ]
    if unblocked:
        lines.append(f"Unblocked non-human-blocked tasks so workers can continue: {', '.join(unblocked)}.")
    if reclaimed:
        lines.append(f"Reclaimed stale/interrupted running tasks: {', '.join(reclaimed)}.")
    if spawned:
        lines.append(f"Started ready tasks in the background: {', '.join(spawned)}.")
    if human:
        lines.append("Smallest unblock questions to bundle if it fits the current reply; preserve hard walls and do not take external action until approved:")
        for task, reason in human[:5]:
            reason_line = " ".join((reason or "").split())[:180]
            lines.append(f"- {task.id} — {task.title}: {reason_line}")
    return "\n".join(lines)


def capture_event_to_kanban(
    event: Any,
    config: GatewayKanbanIntakeConfig,
    *,
    prepared_text: Optional[str] = None,
    session_id: Optional[str] = None,
    session_key: Optional[str] = None,
) -> KanbanIntakeCaptureResult:
    """Create a Kanban card for an actionable inbound gateway message."""
    if not config.enabled:
        return KanbanIntakeCaptureResult(False, reason="disabled")
    if bool(getattr(event, "internal", False)):
        return KanbanIntakeCaptureResult(False, reason="internal")

    source = getattr(event, "source", None)
    if not _source_allowed(source, config):
        return KanbanIntakeCaptureResult(False, reason="source_not_allowed")

    text = prepared_text if prepared_text is not None else getattr(event, "text", "")
    if not should_capture_actionable_task(text):
        return KanbanIntakeCaptureResult(False, reason="not_actionable")

    idempotency_key = _idempotency_key(event, source, session_key, text)

    from hermes_cli import kanban_db

    conn = kanban_db.connect(board=config.board)
    try:
        existing = _existing_task_id(conn, idempotency_key)
        if existing:
            return KanbanIntakeCaptureResult(False, task_id=existing, reason="duplicate")

        task_id = kanban_db.create_task(
            conn,
            title=_title_from_text(text, config.max_title_chars),
            body=_body_from_event(event, source, text, session_id, session_key),
            assignee=config.assignee,
            created_by=config.created_by,
            tenant=config.tenant,
            priority=config.priority,
            idempotency_key=idempotency_key,
            session_id=session_id,
            board=config.board,
        )
        return KanbanIntakeCaptureResult(True, task_id=task_id, reason="created")
    finally:
        conn.close()


def _source_allowed(source: Any, config: GatewayKanbanIntakeConfig) -> bool:
    if source is None:
        return False

    # No allowlist means "trust the gateway's existing authorization layer".
    # When multiple allowlist types are configured, treat them as alternatives
    # rather than cumulative requirements so one config can cover Telegram IDs,
    # WhatsApp display names, and shared household chat IDs at the same time.
    has_allowlist = bool(
        config.allowed_user_ids
        or config.allowed_user_names
        or config.allowed_chat_ids
    )
    if not has_allowlist:
        return True

    if config.allowed_user_ids and str(getattr(source, "user_id", "")) in config.allowed_user_ids:
        return True
    if config.allowed_user_names:
        user_name = str(getattr(source, "user_name", "") or "").casefold()
        allowed_names = {name.casefold() for name in config.allowed_user_names}
        if user_name in allowed_names:
            return True
    if config.allowed_chat_ids and str(getattr(source, "chat_id", "")) in config.allowed_chat_ids:
        return True
    return False


def _existing_task_id(conn: Any, idempotency_key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
        (idempotency_key,),
    ).fetchone()
    return row["id"] if row else None


def _idempotency_key(event: Any, source: Any, session_key: Optional[str], text: str) -> str:
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", "unknown"))
    chat_id = str(getattr(source, "chat_id", "") or "")
    thread_id = str(getattr(source, "thread_id", "") or "")
    message_id = (
        getattr(event, "message_id", None)
        or getattr(source, "message_id", None)
        or _stable_text_fingerprint(session_key, text)
    )
    return f"gateway-intake:{platform}:{chat_id}:{thread_id}:{message_id}"


def _stable_text_fingerprint(session_key: Optional[str], text: str) -> str:
    payload = f"{session_key or ''}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _title_from_text(text: str, max_chars: int) -> str:
    first_line = " ".join((text or "").strip().splitlines()[0:1])
    first_line = re.sub(r"^\[[^\]]+\]\s*", "", first_line).strip()
    first_line = _LEADING_POLITENESS_RE.sub("", first_line).strip()
    first_line = re.sub(r"^(?:to\s+do:|todo:|reminder\s+to|remind\s+me\s+to)\s*", "", first_line, flags=re.IGNORECASE)
    first_line = first_line.strip(" -–—:;,.!") or "Captured household task"
    title = first_line[0].upper() + first_line[1:] if first_line else "Captured household task"
    if len(title) > max_chars:
        title = title[: max_chars - 1].rstrip() + "…"
    return title


def _body_from_event(
    event: Any,
    source: Any,
    text: str,
    session_id: Optional[str],
    session_key: Optional[str],
) -> str:
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", "unknown"))
    burden = _burden_carrier_metadata(source, text)
    lines = [
        "Automatic Kanban capture from gateway chat intake.",
        "",
        "Source:",
        f"source_platform: {platform}",
        f"source_chat_id: {getattr(source, 'chat_id', '') or ''}",
        f"source_chat_name: {getattr(source, 'chat_name', '') or ''}",
        f"source_chat_type: {getattr(source, 'chat_type', '') or ''}",
        f"source_thread_id: {getattr(source, 'thread_id', '') or ''}",
        f"source_message_id: {getattr(event, 'message_id', None) or getattr(source, 'message_id', '') or ''}",
        f"sender_id: {getattr(source, 'user_id', '') or ''}",
        f"sender_name: {getattr(source, 'user_name', '') or ''}",
        f"session_id: {session_id or ''}",
        f"session_key: {session_key or ''}",
        "",
        "Burden-carrier metadata:",
        f"carrier_current: {burden['carrier_current']}",
        f"carrier_target: {burden['carrier_target']}",
        f"relief_priority: {burden['relief_priority']}",
        "invisible_load_type:",
        *[f"  - {load_type}" for load_type in burden["invisible_load_type"]],
        "mental_load_categories:",
        *[f"  - {category}" for category in burden["mental_load_categories"]],
        "",
        "Requested text:",
        text.strip(),
    ]
    return "\n".join(lines).strip() + "\n"


def _burden_carrier_metadata(source: Any, text: str) -> dict[str, Any]:
    """Best-effort burden-carrier fields for automatic Kanban intake.

    Gateway intake runs before the agent has reasoned deeply about the request,
    so this function only records conservative defaults. It names the sender as
    the likely current carrier when the sender is a known household principal;
    otherwise it uses ``unknown`` instead of inventing a carrier. The worker can
    refine these fields in comments/completion metadata after reading the full
    card.
    """

    sender = str(getattr(source, "user_name", "") or "").strip()
    sender_key = sender.casefold()
    if sender_key == "renata":
        carrier_current = "Renata"
    elif sender_key == "zion":
        carrier_current = "Zion"
    else:
        carrier_current = "unknown"

    normalized = f" {text or ''} ".casefold()
    child_related = any(term in normalized for term in (" zaya", " kid", " child", " school", " daycare"))
    product_related = any(term in normalized for term in (" hermes", " gateway", " kanban", " dashboard", " hum product", " product"))

    if carrier_current == "Renata":
        relief_priority = "P0_Renata"
    elif child_related:
        relief_priority = "P1_Zaya"
    elif product_related:
        relief_priority = "P3_product"
    else:
        relief_priority = "P2_household"

    if any(term in normalized for term in (" zaya", " kid", " child", " medical", " doctor", " school")):
        carrier_target = "parent"
    else:
        carrier_target = "Mia"

    load_types: list[str] = ["identifying", "noticing"]
    if re.search(r"\b(anticipate|ahead|upcoming|before|next\s+(?:week|month|season|trip)|transition|expires?|deadline)\b", normalized):
        load_types.append("anticipation")
    if re.search(r"\b(monitor|watch|track|keep\s+an\s+eye|check|status|size|growth)\b", normalized):
        load_types.append("monitoring")
    if re.search(r"\b(remind|remember|reminder|todo|to do|save|album|photo|keepsake|birthday|milestone)\b", normalized):
        load_types.append("remembering")
    if re.search(r"\b(plan|prepare|make|decide|choose|pick|recommend)\b", normalized):
        load_types.append("deciding")
    if re.search(r"\b(find|look\s+into|research|compare|options?|criteria|tradeoffs?)\b", normalized):
        load_types.append("researching")
    if re.search(r"\b(book|schedule|order|buy|get|call|text|send|cancel|reschedule|handle)\b", normalized):
        load_types.append("coordinating")
    if re.search(r"\b(follow\s*up|check|status|chase|reply)\b", normalized):
        load_types.append("following_up")
    if re.search(r"\b(stress|overwhelm|chaos|again|tired|worried|resent|frustrat)\b", normalized):
        load_types.append("emotional_containment")
    if re.search(r"\b(organic|clean|low[-\s]?tox|values?|standard|tradition|family\s+identity|what\s+kind\s+of\s+family)\b", normalized):
        load_types.append("identity_values_work")
    if re.search(r"\b(backup|emergency|urgent|just in case|before it becomes|contingency|pre[-\s]?position)\b", normalized):
        load_types.append("crisis_prepositioning")
    if re.search(r"\b(health|medical|doctor|dentist|medicine|medication|symptom|vaccine|growth|size|shoes?|therapy)\b", normalized):
        load_types.append("health_surveillance")
    if re.search(r"\b(price|cost|budget|bill|invoice|insurance|subscription|renewal|pay|payment|fee|quote)\b", normalized):
        load_types.append("financial_cognitive_load")
    if re.search(r"\b(photo|album|keepsake|memory|milestone|birthday|tradition)\b", normalized):
        load_types.append("memory_keepsake")
    if re.search(r"\b(self[-\s]?care|rest|sleep|workout|exercise|massage|break|protect\s+renata|renata's\s+rest|mom's\s+own\s+needs)\b", normalized):
        load_types.append("carrier_self_care")

    load_types = [item for item in dict.fromkeys(load_types) if item in INVISIBLE_LOAD_TYPES]
    mental_categories = _mental_load_categories_from_types(load_types)

    return {
        "carrier_current": carrier_current,
        "carrier_target": carrier_target,
        "relief_priority": relief_priority,
        "invisible_load_type": load_types,
        "mental_load_categories": mental_categories,
    }


def _mental_load_categories_from_types(load_types: list[str]) -> list[str]:
    categories: list[str] = []
    if any(item in load_types for item in ("coordinating", "following_up", "identifying", "noticing")):
        categories.append("execution")
    if any(item in load_types for item in ("anticipation", "monitoring")):
        categories.append("anticipation")
    if any(item in load_types for item in ("researching", "deciding")):
        categories.append("decision_research")
    if "following_up" in load_types:
        categories.append("relationship_maintenance")
    if "emotional_containment" in load_types:
        categories.append("emotional_labor")
    if "identity_values_work" in load_types:
        categories.append("identity_values_work")
    if "crisis_prepositioning" in load_types:
        categories.append("crisis_prepositioning")
    if "health_surveillance" in load_types:
        categories.append("health_surveillance")
    if "financial_cognitive_load" in load_types:
        categories.append("financial_cognitive_load")
    if any(item in load_types for item in ("memory_keepsake", "carrier_self_care")):
        categories.append("memory_keepsake_carrier_self_care")
    return [item for item in dict.fromkeys(categories) if item in MENTAL_LOAD_CATEGORIES]

# --- HUM-743 pre-turn durable intake API ---
import json as _hum743_json
import os as _hum743_os
import urllib.error as _hum743_urllib_error
import urllib.request as _hum743_urllib_request
from typing import Iterable as _Hum743Iterable

_HUM743_INVISIBLE_LOAD_TYPES = {"anticipation", "monitoring", "coordination", "decision", "emotional_containment", "memory_keepsake", "research", "planning", "standard_setting", "follow_through"}
_HUM743_CARD_TYPES = {"action", "absorbed_load", "decision_with_default", "pattern_to_memorize", "household_fact", "household_preference"}
_HUM743_VISIBILITIES = {"user_visible", "mia_silent", "check_in_later"}
_HUM743_TRUSTED_SOURCE_TYPES = {'user_text', 'voice_transcript', 'screenshot_chat_bubble'}


@dataclass
class IntakeCard:
    card_type: str
    title: str
    body: str
    mental_load_category: Optional[str] = None
    visibility: str = "user_visible"
    proposed_default: Optional[str] = None
    source: str = "regex"


_HUM743_DECLARATIVE_ACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:please\s+)?confirm\s+([^.;!?\n]+)", re.I), "Confirm: {match}"),
    (re.compile(r"\b(?:also\s+)?when\s+is\s+([^.;!?\n]+)", re.I), "Confirm timing: {match}"),
    (re.compile(r"\b(?:and\s+)?what\s+time\s+(?:are\s+)?(?:we|renata|zion|zaya)\s+([^.;!?\n]+)", re.I), "Confirm timing: {match}"),
    (re.compile(r"\bi\s+want\s+to\s+know\s+([^.;!?\n]+)", re.I), "Find out: {match}"),
    (re.compile(r"\bi\s+think\s+([^.;!?\n]+?)\s+will\s+happen\s+([^.;!?\n]+)", re.I), "Track/update: {match}"),
    (re.compile(r"\bi\s+think\s+i\s+found\s+someone\s+to\s+([^.;!?\n]+)", re.I), "Track/update: {match}"),
    (re.compile(r"\bi\s+think\s+renata\s+said\s+she\s+wanted\s+to\s+([^.;!?\n]+)", re.I), "Track/update: {match}"),
    (re.compile(r"\bwe\s+still\s+need\s+to\s+see\s+([^.;!?\n]+)", re.I), "Follow up: {match}"),
    (re.compile(r"\bwe\s+need\s+(?!to\b)([^.;!?\n]+)", re.I), "Handle: {match}"),
    (re.compile(r"\b(?:renata|zion|zaya|we|i)\s+(?:wants?|needs?|has|have)\s+to\s+([^.;!?\n]+)", re.I), "Follow up: {match}"),
    (re.compile(r"\bi\s+need\s+to\s+(?:figure\s+out|sort\s+out|decide|handle)\s+([^.;!?\n]+)", re.I), "Figure out: {match}"),
    (re.compile(r"\bwe\s+should\s+(?:discuss|decide|handle|book|schedule|order|buy|ask|confirm)\s+([^.;!?\n]+)", re.I), "Handle: {match}"),
    (re.compile(r"\bheads\s+up[:,]?\s*([^.;!?\n]+)", re.I), "Heads up: {match}"),
    (re.compile(r"\b(?:figure|sort|carve)\s+out\s+([^.;!?\n]+)", re.I), "Sort out: {match}"),
    (re.compile(r"\b(?:please\s+)?(?:book|schedule|order|buy|call|message|ask|confirm|remind|find)\s+([^.;!?\n]+)", re.I), "Do: {match}"),
)

_HUM743_MENTAL_LOAD_HINTS: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
    (re.compile(r"\bstart packing|pack(?:ing)?\b", re.I), "anticipation", "Absorb packing prep monitoring", "Track the packing timeline and surface only blockers/default decisions."),
    (re.compile(r"\btomorrow|next week|by \w+|before \w+", re.I), "monitoring", "Monitor timing/deadline", "Remember the timing constraint and check back when action is due."),
    (re.compile(r"\brenata wants?|renata needs?|renata asked", re.I), "coordination", "Coordinate around Renata preference", "Treat Renata's preference as context; avoid making Zion restate it."),
    (re.compile(r"\bfigure out|sort out|decide|discuss\b", re.I), "decision", "Carry decision framing", "Prepare a default recommendation unless user vetoes."),
    (re.compile(r"\bworried|stress|overwhelmed|concerned|afraid\b", re.I), "emotional_containment", "Contain emotional load", "Acknowledge the concern without adding planning burden."),
    (re.compile(r"\bremember|keepsake|photo|filming\b", re.I), "memory_keepsake", "Preserve memory/keepsake intent", "Capture the keepsake intent and roll into follow-up/digest."),
    (re.compile(r"\bchef|vendor|vendors|supplier|shop|logistics\b", re.I), "coordination", "Absorb third-party coordination", "Track who needs to be contacted and prepare the next external-facing draft/default."),
    (re.compile(r"\bplan|prep|packing|travel|itinerary|logistics\b", re.I), "planning", "Absorb planning scaffolding", "Turn the loose planning burden into a small ordered checklist before surfacing it."),
    (re.compile(r"\bfigure out|find|research|look into|options?\b", re.I), "research", "Absorb research burden", "Research or frame options internally and return one recommended default if needed."),
    (re.compile(r"\bconfirm|check|follow up|chase|make sure\b", re.I), "follow_through", "Absorb follow-through tracking", "Own the reminder/chase loop and surface only if blocked."),
    (re.compile(r"\bshould|default|standard|usual|preference\b", re.I), "standard_setting", "Capture emerging household standard", "Save the recurring preference/default so the household does not re-specify it next time."),
)

_HUM743_HOUSEHOLD_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\brenata\s+went\s+shopping\b.*\bgot\s+(?:the\s+)?([^.;!?\n]+)", re.I), "Household fact: Renata got {match}", "Acknowledge what is already covered and remember it for the current household loop."),
    (re.compile(r"\b(renata|zion|zaya|mia)\s+(?:got|bought|picked\s+up|ordered|scheduled|booked|paid|sent|saved)\s+(?:the\s+)?([^.;!?\n]+)", re.I), "Household fact: {actor} got {match}", "Acknowledge the new fact and use it instead of asking again."),
)
_HUM743_HOUSEHOLD_PREFERENCE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bi\s+usually\s+need\s+([^.;!?\n]+)", re.I), "Household preference: keep {match} available"),
    (re.compile(r"\bi\s+(?:need|prefer|like)\s+([^.;!?\n]+)\s+(?:for my cut|for cutting|for diet)", re.I), "Household preference: {match}"),
)
_HUM743_FOLLOW_THROUGH_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(?:we\s+would\s+have\s+to|we\s+need\s+to|need\s+to)\s+ask\s+(?:her|renata)\b", re.I), "Ask Renata for the missing household detail", "If this detail matters for the current task, ask Renata; otherwise remember that the fact is unknown."),
)

_HUM743_UI_OCR_FRAGMENT_RE = re.compile(
    r"\b("
    r"bubble|bubbles|composer|delivery status|status line|hourglass|"
    r"visible|whatsapp|screenshot|screen shot|chat bubble|green ones|"
    r"gray/white|grey/white|right side|left side|top of the image|"
    r"bottom of the image|message input|conversation view"
    r")\b",
    re.I,
)
_HUM743_HOUSEHOLD_SIGNAL_RE = re.compile(
    r"\b("
    r"renata|zion|zaya|mia|calendar|party|filming|video shoot|moving|move|"
    r"saturday|wednesday|friday|tomorrow|travel|leaving|packing|chef|"
    r"shopping|yogurt|banana|appointment|dentist|school"
    r")\b",
    re.I,
)


def _hum743_clean_fragment(text: str, limit: int = 120) -> str:
    frag = re.sub(r"\s+", " ", (text or "").strip(" -:;,.\n\t"))
    if len(frag) > limit:
        frag = frag[: limit - 1].rstrip() + "..."
    return frag


def _hum743_format_match(title_template: str, match: re.Match[str]) -> tuple[str, str]:
    if len(match.groups()) >= 2:
        frag = _hum743_clean_fragment(" ".join(group for group in match.groups() if group))
    elif match.groups():
        frag = _hum743_clean_fragment(match.group(1))
    else:
        frag = _hum743_clean_fragment(match.group(0))
    return title_template.format(match=frag), frag


def _hum743_looks_like_ui_ocr_fragment(fragment: str, full_message: str) -> bool:
    if not fragment:
        return True
    if not _HUM743_UI_OCR_FRAGMENT_RE.search(fragment):
        return False
    if _HUM743_HOUSEHOLD_SIGNAL_RE.search(fragment):
        return False
    return bool(_HUM743_UI_OCR_FRAGMENT_RE.search(full_message or ""))


def _hum743_regex_extract(message: str) -> list[IntakeCard]:
    cards: list[IntakeCard] = []
    seen: set[tuple[str, str]] = set()
    action_fragments: list[str] = []
    for pattern, title_template in _HUM743_DECLARATIVE_ACTION_PATTERNS:
        for match in pattern.finditer(message or ""):
            title, frag = _hum743_format_match(title_template, match)
            if not frag:
                continue
            if _hum743_looks_like_ui_ocr_fragment(frag, message or ""):
                continue
            norm_frag = frag.casefold()
            if any(norm_frag in existing or existing in norm_frag for existing in action_fragments):
                continue
            action_fragments.append(norm_frag)
            title = _hum743_clean_fragment(title, 160)
            key = ("action", title.casefold())
            if key in seen:
                continue
            seen.add(key)
            cards.append(IntakeCard(card_type="action", title=title, body=f"Pre-agent regex capture from inbound message: {frag}", visibility="user_visible", proposed_default="Capture and execute or acknowledge this action item."))
    for pattern, title_template, proposed in _HUM743_HOUSEHOLD_FACT_PATTERNS:
        for match in pattern.finditer(message or ""):
            actor = None
            if len(match.groups()) == 2:
                actor = _hum743_clean_fragment(match.group(1), 40).title()
                frag = _hum743_clean_fragment(match.group(2))
            else:
                frag = _hum743_clean_fragment(match.group(1))
            if not frag:
                continue
            title = _hum743_clean_fragment(title_template.format(actor=actor or "Household", match=frag), 160)
            key = ("household_fact", title.casefold())
            if key in seen:
                continue
            seen.add(key)
            cards.append(IntakeCard(card_type="household_fact", title=title, body=f"User supplied household fact before model turn: {frag}", mental_load_category="coordination", visibility="user_visible", proposed_default=proposed))
    for pattern, title_template in _HUM743_HOUSEHOLD_PREFERENCE_PATTERNS:
        for match in pattern.finditer(message or ""):
            frag = _hum743_clean_fragment(match.group(1), 140)
            if not frag:
                continue
            title = _hum743_clean_fragment(title_template.format(match=frag), 160)
            key = ("household_preference", title.casefold())
            if key in seen:
                continue
            seen.add(key)
            cards.append(IntakeCard(card_type="household_preference", title=title, body=f"User supplied recurring household preference before model turn: {frag}", mental_load_category="standard_setting", visibility="user_visible", proposed_default="Acknowledge and save this as a reusable household preference unless it is clearly one-off."))
    for pattern, title, proposed in _HUM743_FOLLOW_THROUGH_FACT_PATTERNS:
        if pattern.search(message or ""):
            key = ("action", title.casefold())
            if key not in seen:
                seen.add(key)
                cards.append(IntakeCard(card_type="action", title=title, body="User indicated a missing fact requires asking Renata.", mental_load_category="follow_through", visibility="user_visible", proposed_default=proposed))
    for pattern, category, title, proposed in _HUM743_MENTAL_LOAD_HINTS:
        if pattern.search(message or ""):
            if category == "standard_setting" and any(card.card_type == "household_preference" for card in cards):
                continue
            key = (category, title.casefold())
            if key in seen:
                continue
            seen.add(key)
            card_type = "decision_with_default" if category == "decision" else "absorbed_load"
            visibility = "check_in_later" if category in {"monitoring", "decision"} else "mia_silent"
            cards.append(IntakeCard(card_type=card_type, title=title, body="Invisible-load capture inferred before model turn.", mental_load_category=category, visibility=visibility, proposed_default=proposed))
    return cards


def _hum743_llm_enabled() -> bool:
    return bool(_hum743_os.environ.get("OPENROUTER_API_KEY") or _hum743_os.environ.get("HERMES_INTAKE_LLM_API_KEY"))


def _hum743_classify_inbound_via_llm(message: str, timeout: float = 0.45) -> list[IntakeCard]:
    api_key = _hum743_os.environ.get("HERMES_INTAKE_LLM_API_KEY") or _hum743_os.environ.get("OPENROUTER_API_KEY")
    if not api_key or not (message or "").strip():
        return []
    model = _hum743_os.environ.get("HERMES_INTAKE_LLM_MODEL", "openai/gpt-4.1-nano")
    prompt = "Extract household action items and invisible mental load from this inbound message. Return ONLY JSON array objects with keys: type(action|load), category, title, body, proposed_action(silent_background|default_with_veto|check_in_later|memorize_pattern), what_principal_is_carrying, what_mia_could_absorb. Categories: " + ", ".join(sorted(_HUM743_INVISIBLE_LOAD_TYPES)) + "\nMessage: " + message[:2000]
    payload = _hum743_json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 700}).encode("utf-8")
    req = _hum743_urllib_request.Request("https://openrouter.ai/api/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with _hum743_urllib_request.urlopen(req, timeout=timeout) as resp:
            data = _hum743_json.loads(resp.read().decode("utf-8"))
        parsed = _hum743_json.loads(data["choices"][0]["message"]["content"])
    except (OSError, _hum743_urllib_error.URLError, KeyError, IndexError, _hum743_json.JSONDecodeError, TimeoutError):
        return []
    cards: list[IntakeCard] = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type") or "").strip().lower()
        proposed_action = str(item.get("proposed_action") or "").strip().lower()
        category = str(item.get("category") or "").strip().lower() or None
        if category not in _HUM743_INVISIBLE_LOAD_TYPES:
            category = None
        if raw_type == "action":
            card_type = "action"; visibility = "user_visible"
        elif proposed_action == "default_with_veto":
            card_type = "decision_with_default"; visibility = "user_visible"
        elif proposed_action == "memorize_pattern":
            card_type = "pattern_to_memorize"; visibility = "mia_silent"
        else:
            card_type = "absorbed_load"; visibility = "check_in_later" if proposed_action == "check_in_later" else "mia_silent"
        title = _hum743_clean_fragment(str(item.get("title") or item.get("what_mia_could_absorb") or raw_type), 160)
        if not title:
            continue
        body_parts = [str(item.get("body") or "").strip()]
        if item.get("what_principal_is_carrying"):
            body_parts.append("Principal carrying: " + str(item["what_principal_is_carrying"]).strip())
        cards.append(IntakeCard(card_type=card_type, mental_load_category=category, title=title, body="\n".join(part for part in body_parts if part) or "Pre-agent LLM intake capture.", visibility=visibility, proposed_default=str(item.get("what_mia_could_absorb") or item.get("proposed_action") or "").strip() or None, source="llm"))
    return cards


def _hum743_source_message_id(session_key: str, event_message_id: Optional[str], message: str) -> str:
    if event_message_id:
        return str(event_message_id)
    h = hashlib.sha256(f"{session_key}\0{message}".encode("utf-8", "ignore")).hexdigest()[:16]
    return f"intake:{h}"


def _hum743_write_card(conn: Any, card: IntakeCard, *, session_key: str, source_message_id: str, provenance_unit: Optional[dict[str, Any]] = None) -> str:
    from hermes_cli import kanban_db as kb
    idempotency = hashlib.sha256(f"kanban-intake\0{source_message_id}\0{card.card_type}\0{card.title}".encode("utf-8", "ignore")).hexdigest()
    metadata = {"source": "gateway.kanban_intake.hum743", "extractor": card.source, "session_key": session_key, "card_type": card.card_type, "mental_load_category": card.mental_load_category, "visibility": card.visibility, "proposed_default": card.proposed_default, "source_message_id": source_message_id}
    if provenance_unit:
        metadata["source_unit"] = {key: provenance_unit.get(key) for key in ("unit_id", "source_message_id", "source_type", "sender_id", "sender_name", "channel", "chat_id", "timestamp") if provenance_unit.get(key) is not None}
    body = (card.body + "\n\n[Intake metadata]\n" + _hum743_json.dumps(metadata, ensure_ascii=False, sort_keys=True)).strip()
    task_id = kb.create_task(conn, title=card.title, body=body, assignee="mia", created_by="gateway-intake", priority=1 if card.card_type == "action" else 0, kind="household", triage=True, idempotency_key=idempotency)
    conn.execute("UPDATE tasks SET card_type = ?, mental_load_category = ?, visibility = ?, proposed_default = ?, source_message_id = ? WHERE id = ?", (card.card_type, card.mental_load_category, card.visibility, card.proposed_default, source_message_id, task_id))
    return task_id


def _hum743_fallback_unit(message: str, *, session_key: str, event_message_id: Optional[str]) -> dict[str, Any]:
    source_message_id = _hum743_source_message_id(session_key or "", event_message_id, message)
    return {"unit_id": source_message_id, "source_message_id": source_message_id, "source_type": "user_text", "trusted_for_intake": True, "text": message}


def _hum743_normalize_provenance_units(message: str, provenance_units: Optional[_Hum743Iterable[dict[str, Any]]], *, session_key: str, event_message_id: Optional[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    provided_units = provenance_units is not None
    for raw in provenance_units or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        source_type = str(raw.get("source_type") or "unknown").strip() or "unknown"
        trusted = raw.get("trusted_for_intake")
        if trusted is None:
            trusted = source_type in _HUM743_TRUSTED_SOURCE_TYPES
        if not trusted:
            continue
        unit = dict(raw)
        unit["text"] = text
        unit["source_type"] = source_type
        unit["trusted_for_intake"] = True
        unit_id = str(unit.get("unit_id") or unit.get("source_message_id") or "").strip()
        source_message_id = str(unit.get("source_message_id") or unit_id or "").strip()
        if not source_message_id:
            source_message_id = _hum743_source_message_id(session_key or "", event_message_id, text)
        unit["source_message_id"] = source_message_id
        unit["unit_id"] = unit_id or source_message_id
        normalized.append(unit)
    if normalized:
        return normalized
    if provided_units:
        return []
    return [_hum743_fallback_unit(message, session_key=session_key, event_message_id=event_message_id)]


def capture_inbound(message: str, *, session_key: str = "", event_message_id: Optional[str] = None, provenance_units: Optional[_Hum743Iterable[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    msg = message if isinstance(message, str) else str(message or "")
    from hermes_cli import kanban_db as kb
    captured: list[dict[str, Any]] = []
    units = _hum743_normalize_provenance_units(msg, provenance_units, session_key=session_key or "", event_message_id=event_message_id)
    with kb.connect() as conn:
        for unit in units:
            unit_text = str(unit.get("text") or "")
            cards = _hum743_regex_extract(unit_text)
            if _hum743_os.environ.get("HERMES_INTAKE_LLM_ALWAYS") == "1" or (not cards and _hum743_llm_enabled()):
                cards.extend(_hum743_classify_inbound_via_llm(unit_text))
            if not cards:
                continue
            src_id = str(unit.get("source_message_id") or _hum743_source_message_id(session_key or "", event_message_id, unit_text))
            for card in cards:
                if card.card_type not in _HUM743_CARD_TYPES:
                    continue
                if card.visibility not in _HUM743_VISIBILITIES:
                    card.visibility = "user_visible"
                if card.mental_load_category not in _HUM743_INVISIBLE_LOAD_TYPES:
                    card.mental_load_category = None
                task_id = _hum743_write_card(conn, card, session_key=session_key or "", source_message_id=src_id, provenance_unit=unit)
                captured.append({"id": task_id, "card_type": card.card_type, "mental_load_category": card.mental_load_category, "visibility": card.visibility, "title": card.title, "proposed_default": card.proposed_default, "source_message_id": src_id, "source_unit_id": unit.get("unit_id") or src_id, "source_type": unit.get("source_type")})
    return captured


def format_pre_turn_context(cards: _Hum743Iterable[dict[str, Any]]) -> str:
    cards = list(cards or [])
    if not cards:
        return ""
    action_count = sum(1 for card in cards if card.get("card_type") == "action")
    fact_count = sum(1 for card in cards if card.get("card_type") == "household_fact")
    preference_count = sum(1 for card in cards if card.get("card_type") == "household_preference")
    load_count = len(cards) - action_count - fact_count - preference_count
    lines = [f"Pre-turn intake: {action_count} action cards + {fact_count} household fact cards + {preference_count} preference cards + {load_count} mental load cards were captured BEFORE this turn.", "Your job is to execute / acknowledge / default-question / silently-absorb each. User-visible household facts/preferences require a brief acknowledgment and should be saved or used instead of asking again. Do NOT re-discover them; act on them.", "Cards:"]
    for card in cards[:30]:
        bits = [str(card.get("id") or "?"), str(card.get("card_type") or "?"), str(card.get("visibility") or "?")]
        if card.get("mental_load_category"):
            bits.append(str(card["mental_load_category"]))
        lines.append(f"- {' | '.join(bits)}: {card.get('title')}")
        if card.get("proposed_default"):
            lines.append(f"  proposed_default: {card.get('proposed_default')}")
    return "\n".join(lines)
