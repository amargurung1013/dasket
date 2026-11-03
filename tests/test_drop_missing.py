import polars as pl

from datum import Dataset


def test_drop_missing_removes_rows_with_nulls():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", None, "Sorting"],
            "language": ["python", "go", "python"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    cleaned = dataset.drop_missing()

    assert cleaned.row_count == 2
    assert cleaned.data["title"].to_list() == [
        "Binary Search",
        "Sorting",
    ]


def test_drop_missing_uses_selected_columns():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Sorting", "Graphs"],
            "description": ["Search algorithm", None, "Graph traversal"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    cleaned = dataset.drop_missing(columns=["title"])

    assert cleaned.row_count == 3


def test_drop_missing_preserves_original():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", None],
        }
    )
    dataset = Dataset.from_dataframe(df)

    cleaned = dataset.drop_missing()

    assert dataset.row_count == 2
    assert cleaned.row_count == 1


def test_drop_missing_returns_same_data_when_no_nulls():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Sorting"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    cleaned = dataset.drop_missing()

    assert cleaned.row_count == 2


def test_drop_missing_handles_empty_dataset():
    df = pl.DataFrame(
        {
            "title": pl.Series([], dtype=pl.String),
        }
    )
    dataset = Dataset.from_dataframe(df)

    cleaned = dataset.drop_missing()

    assert cleaned.row_count == 0
