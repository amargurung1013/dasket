import polars as pl
import pytest

from datum import Dataset


@pytest.fixture
def dataset():
    df = pl.DataFrame(
        {
            "id": list(range(100)),
            "label": ["positive"] * 80 + ["negative"] * 20,
        }
    )
    return Dataset.from_dataframe(df)


def test_stratified_split_preserves_class_distribution(dataset):
    train, test = dataset.stratified_split(
        column="label",
        train_ratio=0.8,
        seed=42,
    )

    assert train.row_count == 80
    assert test.row_count == 20

    train_counts = dict(train.data["label"].value_counts().iter_rows())
    test_counts = dict(test.data["label"].value_counts().iter_rows())

    assert train_counts == {"positive": 64, "negative": 16}
    assert test_counts == {"positive": 16, "negative": 4}


def test_stratified_split_is_reproducible(dataset):
    train1, test1 = dataset.stratified_split("label", seed=42)
    train2, test2 = dataset.stratified_split("label", seed=42)

    assert train1.data.equals(train2.data)
    assert test1.data.equals(test2.data)


def test_stratified_split_has_no_overlapping_rows(dataset):
    train, test = dataset.stratified_split("label", seed=42)

    train_ids = set(train.data["id"].to_list())
    test_ids = set(test.data["id"].to_list())

    assert train_ids.isdisjoint(test_ids)


def test_stratified_split_preserves_original(dataset):
    dataset.stratified_split("label", seed=42)

    assert dataset.row_count == 100


def test_stratified_split_rejects_missing_column(dataset):
    with pytest.raises(ValueError, match="not found"):
        dataset.stratified_split("unknown")


def test_stratified_split_rejects_invalid_ratio(dataset):
    with pytest.raises(ValueError, match="train_ratio"):
        dataset.stratified_split("label", train_ratio=1.5)


def test_stratified_split_rejects_classes_with_insufficient_samples():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "label": ["A", "A", "B"],
        }
    )
    dataset = Dataset.from_dataframe(df)

    with pytest.raises(ValueError, match="at least 2"):
        dataset.stratified_split("label")
