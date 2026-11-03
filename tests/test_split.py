import polars as pl
import pytest

from dasket import Dataset


@pytest.fixture
def dataset():
    df = pl.DataFrame(
        {
            "id": list(range(100)),
            "title": [f"Task {i}" for i in range(100)],
        }
    )
    return Dataset.from_dataframe(df)


def test_split_respects_train_ratio(dataset):
    train, test = dataset.split(train_ratio=0.8, seed=42)

    assert train.row_count == 80
    assert test.row_count == 20


def test_split_preserves_all_rows(dataset):
    train, test = dataset.split(train_ratio=0.8, seed=42)

    assert train.row_count + test.row_count == dataset.row_count


def test_split_is_reproducible(dataset):
    train1, test1 = dataset.split(train_ratio=0.8, seed=42)
    train2, test2 = dataset.split(train_ratio=0.8, seed=42)

    assert train1.data.equals(train2.data)
    assert test1.data.equals(test2.data)


def test_split_has_no_overlapping_rows(dataset):
    train, test = dataset.split(train_ratio=0.8, seed=42)

    train_ids = set(train.data["id"].to_list())
    test_ids = set(test.data["id"].to_list())

    assert train_ids.isdisjoint(test_ids)


def test_split_does_not_modify_original(dataset):
    dataset.split(train_ratio=0.8, seed=42)

    assert dataset.row_count == 100


def test_split_rejects_invalid_ratio(dataset):
    with pytest.raises(ValueError, match="train_ratio"):
        dataset.split(train_ratio=1.5)


def test_split_rejects_zero_ratio(dataset):
    with pytest.raises(ValueError, match="train_ratio"):
        dataset.split(train_ratio=0)


def test_split_handles_empty_dataset():
    df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
        }
    )
    dataset = Dataset.from_dataframe(df)

    train, test = dataset.split(train_ratio=0.8, seed=42)

    assert train.row_count == 0
    assert test.row_count == 0
