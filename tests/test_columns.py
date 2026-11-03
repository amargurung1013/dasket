import polars as pl
import pytest

from datum import Dataset


@pytest.fixture
def dataset():
    df = pl.DataFrame(
        {
            "title": ["Binary Search", "Sorting"],
            "language": ["Python", "Go"],
            "difficulty": ["Medium", "Easy"],
        }
    )
    return Dataset.from_dataframe(df)


def test_select_columns(dataset):
    result = dataset.select_columns(["title", "language"])

    assert result.columns == ["title", "language"]
    assert result.row_count == 2


def test_drop_columns(dataset):
    result = dataset.drop_columns(["difficulty"])

    assert result.columns == ["title", "language"]


def test_rename_columns(dataset):
    result = dataset.rename_columns({"title": "task_title"})

    assert "task_title" in result.columns
    assert "title" not in result.columns


def test_column_operations_preserve_original(dataset):
    dataset.select_columns(["title"])
    dataset.drop_columns(["language"])
    dataset.rename_columns({"title": "task_title"})

    assert dataset.columns == ["title", "language", "difficulty"]


def test_select_columns_rejects_missing_column(dataset):
    with pytest.raises(ValueError, match="not found"):
        dataset.select_columns(["description"])


def test_drop_columns_rejects_missing_column(dataset):
    with pytest.raises(ValueError, match="not found"):
        dataset.drop_columns(["description"])


def test_rename_columns_rejects_missing_column(dataset):
    with pytest.raises(ValueError, match="not found"):
        dataset.rename_columns({"description": "details"})
