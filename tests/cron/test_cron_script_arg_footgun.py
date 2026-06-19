"""Regression tests for the cron 'script' args-in-field footgun (HUM-1498).

The scheduler treats the entire ``script`` field as a single file path —
``_run_job_script`` builds ``argv = [interpreter, path]`` and parses no CLI
args. A value like ``"mia_signal_loop.py --apply"`` therefore resolves a
literal filename with a space and fails ``Script not found`` silently every
run (the root cause of HUM-1493/1495).

These tests lock in two guardrails:

* ``create_job`` / ``update_job`` reject a ``script`` carrying whitespace or
  flag tokens at config time, with a message pointing at the wrapper fix.
* The runtime ``Script not found`` message hints at the footgun when the
  offending path contains whitespace/flags.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# create_job / update_job: reject args-in-script at config time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_script",
    [
        "mia_signal_loop.py --apply",  # the exact HUM-1495 value
        "watchdog.sh --verbose",
        "foo.py bar baz",  # multiple positional tokens
        "--apply",  # bare flag token, no path
    ],
)
def test_create_job_rejects_script_with_args(hermes_env, bad_script):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="cannot carry CLI arguments or flags"):
        create_job(
            prompt=None,
            schedule="every 5m",
            script=bad_script,
            no_agent=True,
            deliver="local",
        )


def test_create_job_accepts_clean_script_path(hermes_env):
    from cron.jobs import create_job

    (hermes_env / "scripts" / "mia_signal_loop_apply.sh").write_text("echo ok\n")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="mia_signal_loop_apply.sh",
        no_agent=True,
        deliver="local",
    )
    assert job["script"] == "mia_signal_loop_apply.sh"


def test_update_job_rejects_script_with_args(hermes_env):
    from cron.jobs import create_job, update_job

    (hermes_env / "scripts" / "w.sh").write_text("echo ok\n")
    job = create_job(
        prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local"
    )

    with pytest.raises(ValueError, match="cannot carry CLI arguments or flags"):
        update_job(job["id"], {"script": "w.sh --apply"})


# ---------------------------------------------------------------------------
# _run_job_script: runtime "not found" message hints at the footgun
# ---------------------------------------------------------------------------


def test_script_not_found_hints_at_args_footgun(hermes_env):
    """A missing path that carries whitespace/flags gets the diagnostic hint."""
    from cron.scheduler import _run_job_script

    ok, output = _run_job_script("mia_signal_loop.py --apply")
    assert ok is False
    assert "Script not found" in output
    assert "wrapper" in output
    assert "--apply" in output


def test_script_not_found_plain_when_no_args(hermes_env):
    """A plain missing path does not get the args hint (no false positive)."""
    from cron.scheduler import _run_job_script

    ok, output = _run_job_script("nope.py")
    assert ok is False
    assert "Script not found" in output
    assert "wrapper" not in output
