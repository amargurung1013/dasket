import polars as pl
import pytest

from datum import Dataset


def test_normalize_strips_whitespace():
    df = pl.DataFrame({"title": ["  Binary Search  ", "  Sorting  "]})
    dataset = Dataset.from_dataframe(df)

    normalized = dataset.normalize(columns=["title"])

    assert normalized.data["title"].to_list() == [
        "Binary Search",
        "Sorting",
    ]


def test_normalize_lowercases_selected_columns():
    df = pl.DataFrame(
        {
            "title": ["  BINARY SEARCH  "],
            "language": ["  PYTHON  "],
        }
    )
    dataset = Dataset.from_dataframe(df)

    normalized = dataset.normalize(
        columns=["title", "language"],
        lowercase=["title"],
    )

    assert normalized.data["title"].to_list() == ["binary search"]
    assert normalized.data["language"].to_list() == ["PYTHON"]


def test_normalize_preserves_null_values():
    df = pl.DataFrame({"title": ["  Binary Search  ", None]})
    dataset = Dataset.from_dataframe(df)

    normalized = dataset.normalize(columns=["title"])

    assert normalized.data["title"].to_list() == ["Binary Search", None]


def test_normalize_does_not_modify_original():
    df = pl.DataFrame({"title": ["  Binary Search  "]})
    dataset = Dataset.from_dataframe(df)

    normalized = dataset.normalize(columns=["title"])

    assert dataset.data["title"].to_list() == ["  Binary Search  "]
    assert normalized.data["title"].to_list() == ["Binary Search"]


def test_normalize_rejects_missing_columns():
    dataset = Dataset.from_dataframe(pl.DataFrame({"title": ["Hello"]}))

    with pytest.raises(ValueError, match="not found"):
        dataset.normalize(columns=["description"])


def test_normalize_rejects_non_string_columns():
    dataset = Dataset.from_dataframe(pl.DataFrame({"score": [10, 20]}))

    with pytest.raises(TypeError, match="String"):
        dataset.normalize(columns=["score"])


def test_normalize_rejects_lowercase_columns_not_selected():
    dataset = Dataset.from_dataframe(pl.DataFrame({"title": ["Hello"]}))

    with pytest.raises(ValueError, match="must also be in columns"):
        dataset.normalize(columns=["title"], lowercase=["description"])
