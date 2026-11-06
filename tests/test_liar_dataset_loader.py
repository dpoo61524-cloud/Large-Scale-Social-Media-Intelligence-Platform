"""Unit tests for LIAR dataset loader and label normalizer."""

from pathlib import Path
import json
import pytest

from src.misinformation.dataset_loader import (
    FINE_GRAINED_LABEL_TO_ID,
    LiarDatasetLoader,
)


def test_embedded_benchmark_sample_structure() -> None:
    loader = LiarDatasetLoader()
    sample = loader.load_benchmark_sample()
    assert len(sample) >= 20

    for item in sample:
        assert "statement" in item
        assert "label" in item
        assert "speaker" in item
        assert "context" in item
        assert item["label"] in FINE_GRAINED_LABEL_TO_ID


def test_label_normalization() -> None:
    loader = LiarDatasetLoader()
    assert loader.normalize_label("pants-on-fire") == "pants-fire"
    assert loader.normalize_label("Pants-Fire") == "pants-fire"
    assert loader.normalize_label("barely") == "barely-true"
    assert loader.normalize_label("mostly") == "mostly-true"
    assert loader.normalize_label("half") == "half-true"
    assert loader.normalize_label("  TRUE  ") == "true"
    assert loader.normalize_label("FALSE") == "false"


def test_binary_mapping_and_uncertain_policies() -> None:
    loader = LiarDatasetLoader()

    # Fake targets
    assert loader.map_to_binary("pants-fire") == 0
    assert loader.map_to_binary("false") == 0
    assert loader.map_to_binary("barely-true") == 0

    # Real targets
    assert loader.map_to_binary("mostly-true") == 1
    assert loader.map_to_binary("true") == 1

    # Uncertain / half-true policy
    assert loader.map_to_binary("half-true", uncertain_policy="exclude") is None
    assert loader.map_to_binary("half-true", uncertain_policy="fake") == 0
    assert loader.map_to_binary("half-true", uncertain_policy="real") == 1

    # Unknown label
    assert loader.map_to_binary("unknown-label") is None


def test_fine_grained_mapping() -> None:
    loader = LiarDatasetLoader()
    assert loader.map_to_fine_grained("pants-fire") == 0
    assert loader.map_to_fine_grained("false") == 1
    assert loader.map_to_fine_grained("barely-true") == 2
    assert loader.map_to_fine_grained("half-true") == 3
    assert loader.map_to_fine_grained("mostly-true") == 4
    assert loader.map_to_fine_grained("true") == 5
    assert loader.map_to_fine_grained("invalid") is None


def test_prepare_data_binary_and_multiclass() -> None:
    loader = LiarDatasetLoader()

    # Binary mode
    texts_b, labels_b, meta_b = loader.prepare_data(binary=True, uncertain_policy="exclude")
    assert len(texts_b) == len(labels_b) == len(meta_b)
    assert set(labels_b).issubset({0, 1})
    assert all(m["target_type"] == "binary" for m in meta_b)

    # Multi-class mode
    texts_m, labels_m, meta_m = loader.prepare_data(binary=False)
    assert len(texts_m) == len(labels_m) == len(meta_m)
    assert set(labels_m).issubset({0, 1, 2, 3, 4, 5})
    assert all(m["target_type"] == "fine_grained" for m in meta_m)


def test_load_from_tsv(tmp_path: Path) -> None:
    tsv_content = (
        "statement\tlabel\tspeaker\tcontext\n"
        "Claim A is completely false.\tfalse\tPolitician A\tSpeech\n"
        "Claim B is perfectly true.\ttrue\tScientist B\tArticle\n"
    )
    tsv_file = tmp_path / "test_data.tsv"
    tsv_file.write_text(tsv_content, encoding="utf-8")

    loader = LiarDatasetLoader(data_path=tsv_file)
    records = loader.load_from_tsv(tsv_file)
    assert len(records) == 2
    assert records[0]["statement"] == "Claim A is completely false."
    assert records[0]["label"] == "false"
    assert records[1]["statement"] == "Claim B is perfectly true."
    assert records[1]["label"] == "true"


def test_load_from_json(tmp_path: Path) -> None:
    json_data = [
        {"statement": "Claim 1", "label": "mostly-true", "speaker": "Speaker 1"},
        {"statement": "Claim 2", "label": "pants-fire", "speaker": "Speaker 2"},
    ]
    json_file = tmp_path / "test_data.json"
    json_file.write_text(json.dumps(json_data), encoding="utf-8")

    loader = LiarDatasetLoader()
    records = loader.load_from_json(json_file)
    assert len(records) == 2
    assert records[0]["label"] == "mostly-true"
    assert records[1]["label"] == "pants-fire"


def test_missing_file_raises_error(tmp_path: Path) -> None:
    loader = LiarDatasetLoader()
    missing = tmp_path / "non_existent.tsv"
    with pytest.raises(FileNotFoundError):
        loader.load_from_tsv(missing)
    with pytest.raises(FileNotFoundError):
        loader.load_from_json(tmp_path / "non_existent.json")
