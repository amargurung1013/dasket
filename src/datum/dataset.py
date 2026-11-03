import math
import random
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from datum.reports import ValidationReport
from datum.validation import validate_records


class Dataset:
    def __init__(self, data: pl.DataFrame):
        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be a Polars DataFrame")

        self._data = data

    def validate(self, schema: type[BaseModel]) -> ValidationReport:
        result = validate_records(self._data, schema)

        return ValidationReport(**result)

    def normalize(
        self,
        columns: list[str],
        lowercase: list[str] | None = None,
    ) -> "Dataset":
        lowercase = lowercase or []

        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        invalid_lowercase = set(lowercase) - set(columns)

        if invalid_lowercase:
            raise ValueError(
                "Columns in lowercase must also be in columns: "
                f"{sorted(invalid_lowercase)}"
            )

        expressions = []

        for column in columns:
            if self._data.schema[column] != pl.String:
                raise TypeError(f"Column '{column}' must have String dtype")

            expression = pl.col(column).str.strip_chars()

            if column in lowercase:
                expression = expression.str.to_lowercase()

            expressions.append(expression.alias(column))

        normalized_data = self._data.with_columns(expressions)

        return Dataset(normalized_data)

    def find_duplicates(
        self,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        if columns is None:
            columns = self._data.columns

        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        return self._data.filter(pl.struct(columns).is_duplicated()).unique(
            subset=columns, keep="first"
        )

    def missing_values(self) -> dict[str, int]:
        return {
            column: self._data[column].null_count() for column in self._data.columns
        }

    def drop_missing(
        self,
        columns: list[str] | None = None,
    ) -> "Dataset":
        if columns is None:
            columns = self._data.columns

        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        cleaned_data = self._data.drop_nulls(subset=columns)

        return Dataset(cleaned_data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Dataset":
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        suffix = file_path.suffix.lower()

        loaders = {
            ".csv": pl.read_csv,
            ".json": pl.read_json,
            ".jsonl": pl.read_ndjson,
            ".parquet": pl.read_parquet,
        }

        loader = loaders.get(suffix)

        if loader is None:
            raise ValueError(f"Unsupported file format: {suffix}")

        return cls(loader(file_path))

    def profile(self) -> dict:
        return {
            "row_count": self.row_count,
            "column_count": len(self.columns),
            "columns": {
                column: {
                    "dtype": str(self._data.schema[column]),
                    "null_count": self._data[column].null_count(),
                }
                for column in self.columns
            },
            "duplicate_count": self._data.is_duplicated().sum(),
        }

    def to_file(self, path: str | Path) -> None:
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        exporters = {
            ".csv": self._data.write_csv,
            ".json": self._data.write_json,
            ".jsonl": self._data.write_ndjson,
            ".parquet": self._data.write_parquet,
        }

        exporter = exporters.get(suffix)

        if exporter is None:
            raise ValueError(f"Unsupported file format: {suffix}")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        exporter(file_path)

    def select_columns(self, columns: list[str]) -> "Dataset":
        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        return Dataset(self._data.select(columns))

    def drop_columns(self, columns: list[str]) -> "Dataset":
        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        return Dataset(self._data.drop(columns))

    def rename_columns(self, mapping: dict[str, str]) -> "Dataset":
        missing_columns = set(mapping) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        return Dataset(self._data.rename(mapping))

    def filter(self, expression: pl.Expr) -> "Dataset":
        if not isinstance(expression, pl.Expr):
            raise TypeError("expression must be a Polars Expr")

        filtered_data = self._data.filter(expression)

        return Dataset(filtered_data)

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int | None = None,
    ) -> tuple["Dataset", "Dataset"]:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        row_count = self.row_count
        train_count = math.floor(row_count * train_ratio)

        indices = list(range(row_count))
        random.Random(seed).shuffle(indices)

        train_indices = indices[:train_count]
        test_indices = indices[train_count:]

        train_data = self._data[train_indices]
        test_data = self._data[test_indices]

        return Dataset(train_data), Dataset(test_data)

    @classmethod
    def from_dataframe(cls, dataframe: pl.DataFrame) -> "Dataset":
        return cls(dataframe)

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def columns(self) -> list[str]:
        return self._data.columns

    @property
    def row_count(self) -> int:
        return self._data.height
