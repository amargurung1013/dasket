import polars as pl
import pytest

from dasket import Dataset


def test_find_exact_duplicate_rows():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Sorting", "Binary Search"],
            "language": ["python", "go", "python"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    duplicates = dataset.find_duplicates()

    assert duplicates.height == 1
    assert duplicates["title"].to_list() == ["Binary Search"]


def test_find_duplicates_using_selected_columns():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Binary Search", "Sorting"],
            "language": ["python", "go", "python"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    duplicates = dataset.find_duplicates(columns=["title"])

    assert duplicates.height == 1
    assert duplicates["title"].to_list() == ["Binary Search"]


def test_find_duplicates_returns_empty_dataframe_when_none_exist():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Sorting"],
            "language": ["python", "go"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    duplicates = dataset.find_duplicates()

    assert duplicates.is_empty()


def test_find_duplicates_rejects_missing_columns():
    dataset = Dataset.from_dataframe(pl.DataFrame({"title": ["Binary Search"]}))

    with pytest.raises(ValueError, match="not found"):
        dataset.find_duplicates(columns=["description"])


def test_find_duplicates_does_not_modify_original():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Binary Search"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    dataset.find_duplicates()

    assert dataset.row_count == 2
