import polars as pl
import pytest

from dasket import Dataset


def test_create_dataset_from_dataframe():
    df = pl.DataFrame({"name": ["Alice", "Bob"]})

    dataset = Dataset.from_dataframe(df)

    assert dataset.row_count == 2
    assert dataset.columns == ["name"]


def test_reject_invalid_data():
    with pytest.raises(TypeError):
        Dataset("data.csv")


def test_load_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("name\nAlice\nBob\n")

    dataset = Dataset.from_file(path)

    assert dataset.row_count == 2


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        Dataset.from_file("missing.parquet")


def test_unsupported_format(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello")

    with pytest.raises(ValueError):
        Dataset.from_file(path)
