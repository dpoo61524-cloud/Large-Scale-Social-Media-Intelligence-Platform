"""Basic validation checks for main application entry point."""

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
