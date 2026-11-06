"""Basic validation checks for main application entry point."""

import sys
from app import main


def test_app_main_runs_without_exceptions(capsys):
    """Verify app.py main() executes cleanly and outputs expected information."""
    main()
    captured = capsys.readouterr()

    # Verify standard project headers and sections exist
    assert "LARGE SCALE SOCIAL MEDIA INTELLIGENCE PLATFORM" in captured.out
    assert "Block 1 - Project Foundation Initialized" in captured.out
    assert "[PROJECT OBJECTIVES]" in captured.out
    assert "[ENVIRONMENT & STORAGE STATUS]" in captured.out


def test_app_main_with_ingest_flag(capsys, monkeypatch):
    """Verify app.py runs mock Reddit ingestion when --ingest-reddit is passed."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--ingest-reddit"])
    main()
    captured = capsys.readouterr()

    assert "STARTING REDDIT MOCK INGESTION" in captured.out
    assert "Total Records" in captured.out
    assert "Valid Records" in captured.out


def test_app_main_with_live_reddit_flag(capsys, monkeypatch):
    """Verify app.py handles live Reddit ingestion flag gracefully when unconfigured."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--ingest-reddit-live"])
    main()
    captured = capsys.readouterr()

    assert "STARTING REDDIT LIVE API INGESTION" in captured.out
    assert "Reddit Live API" in captured.out


def test_app_main_with_youtube_flag(capsys, monkeypatch):
    """Verify app.py handles live YouTube ingestion flag gracefully when unconfigured."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--ingest-youtube"])
    main()
    captured = capsys.readouterr()

    assert "STARTING YOUTUBE DATA API INGESTION" in captured.out
    assert "YouTube Data API v3" in captured.out


def test_app_main_with_preprocess_reddit_flag(capsys, monkeypatch):
    """Verify app.py handles Reddit preprocessing flag."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--preprocess-reddit"])
    main()
    captured = capsys.readouterr()

    assert "STARTING REDDIT TEXT PREPROCESSING (STEP 2)" in captured.out
    assert "Reddit Posts" in captured.out


def test_app_main_with_preprocess_youtube_flag(capsys, monkeypatch):
    """Verify app.py handles YouTube preprocessing flag."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--preprocess-youtube"])
    main()
    captured = capsys.readouterr()

    assert "STARTING YOUTUBE TEXT PREPROCESSING (STEP 2)" in captured.out
    assert "YouTube Videos" in captured.out
