import polars as pl

from dasket import Dataset


def test_missing_values_counts_nulls_per_column():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", None, "Sorting"],
            "description": [None, None, "An algorithm"],
            "language": ["python", "go", "python"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    result = dataset.missing_values()

    assert result == {
        "title": 1,
        "description": 2,
        "language": 0,
    }


def test_missing_values_returns_zero_for_complete_columns():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Sorting"],
            "language": ["python", "go"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    result = dataset.missing_values()

    assert result == {
        "title": 0,
        "language": 0,
    }


def test_missing_values_handles_empty_dataset():
    df = pl.DataFrame(
        {
            "title": pl.Series([], dtype=pl.String),
            "language": pl.Series([], dtype=pl.String),
        }
    )
    dataset = Dataset.from_dataframe(df)

    result = dataset.missing_values()

    assert result == {
        "title": 0,
        "language": 0,
    }


def test_missing_values_does_not_modify_original():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", None],
        }
    )
    dataset = Dataset.from_dataframe(df)

    dataset.missing_values()

    assert dataset.row_count == 2
    assert dataset.data["title"].null_count() == 1
