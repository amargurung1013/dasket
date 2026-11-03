import polars as pl
import pytest

from dasket import Dataset


@pytest.fixture
def dataset():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Sorting", "Graph Traversal"],
            "language": ["Python", "Go", "Python"],
            "difficulty": ["Medium", "Easy", "Hard"],
        }
    )
    return Dataset.from_dataframe(df)


def test_filter_by_language(dataset):
    result = dataset.filter(pl.col("language") == "Python")

    assert result.row_count == 2
    assert result.data["title"].to_list() == [
        "Binary Search",
        "Graph Traversal",
    ]


def test_filter_by_multiple_conditions(dataset):
    result = dataset.filter(
        (pl.col("language") == "Python") & (pl.col("difficulty") == "Medium")
    )

    assert result.row_count == 1
    assert result.data["title"].to_list() == ["Binary Search"]


def test_filter_returns_empty_dataset_when_no_match(dataset):
    result = dataset.filter(pl.col("language") == "Rust")

    assert result.row_count == 0
    assert result.columns == dataset.columns


def test_filter_does_not_modify_original(dataset):
    dataset.filter(pl.col("language") == "Python")

    assert dataset.row_count == 3


def test_filter_rejects_invalid_expression(dataset):
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        dataset.filter(pl.col("unknown") == "value")
