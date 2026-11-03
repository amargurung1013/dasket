import polars as pl
from pydantic import BaseModel, Field

from datum import Dataset


class CodingTask(BaseModel):
    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)


def test_validate_valid_records():
    df = pl.DataFrame(
        {
            "task_id": ["task-001"],
            "title": ["Binary Search"],
            "language": ["python"],
            "difficulty": ["medium"],
        }
    )

    dataset = Dataset.from_dataframe(df)
    report = dataset.validate(CodingTask)

    assert report.total_count == 1
    assert report.valid_count == 1
    assert report.invalid_count == 0
    assert report.is_valid is True


def test_validate_invalid_records():
    df = pl.DataFrame(
        {
            "task_id": ["task-001"],
            "title": [""],
            "language": ["python"],
            "difficulty": ["medium"],
        }
    )

    dataset = Dataset.from_dataframe(df)
    report = dataset.validate(CodingTask)

    assert report.valid_count == 0
    assert report.invalid_count == 1
    assert report.is_valid is False


def test_validate_missing_fields():
    df = pl.DataFrame(
        {
            "task_id": ["task-001"],
            "title": ["Binary Search"],
        }
    )

    dataset = Dataset.from_dataframe(df)
    report = dataset.validate(CodingTask)

    assert report.invalid_count == 1
    assert len(report.invalid_records[0]["errors"]) == 2


def test_validate_empty_dataset():
    df = pl.DataFrame(
        {
            "task_id": pl.Series([], dtype=pl.String),
            "title": pl.Series([], dtype=pl.String),
            "language": pl.Series([], dtype=pl.String),
            "difficulty": pl.Series([], dtype=pl.String),
        }
    )

    dataset = Dataset.from_dataframe(df)
    report = dataset.validate(CodingTask)

    assert report.total_count == 0
    assert report.valid_count == 0
    assert report.invalid_count == 0
    assert report.is_valid is True
