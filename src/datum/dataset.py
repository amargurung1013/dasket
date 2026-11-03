import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

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

        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates")

        if len(lowercase) != len(set(lowercase)):
            raise ValueError("lowercase must not contain duplicates")

        if not columns:
            return Dataset(self._data.clone())

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

        return Dataset(self._data.with_columns(expressions))

    def find_duplicates(
        self,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        if columns is None:
            columns = self._data.columns

        if not columns:
            raise ValueError("At least one column is required for duplicate detection")

        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates")

        return self._data.filter(pl.struct(columns).is_duplicated()).unique(
            subset=columns,
            keep="first",
            maintain_order=True,
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

        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates")

        if not columns:
            return Dataset(self._data.clone())

        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        cleaned_data = self._data.drop_nulls(subset=columns)

        return Dataset(cleaned_data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Dataset":
        file_path = Path(path)

        if not file_path.is_file():
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

    def profile(self) -> dict[str, Any]:
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
            "duplicate_count": int(self._data.is_duplicated().sum()),
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

        if file_path.exists() and file_path.is_dir():
            raise IsADirectoryError(f"Expected a file path, got directory: {file_path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        exporter(file_path)

    def select_columns(self, columns: list[str]) -> "Dataset":
        if not columns:
            raise ValueError("At least one column must be selected")

        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates")

        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        return Dataset(self._data.select(columns))

    def drop_columns(self, columns: list[str]) -> "Dataset":
        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates")

        missing_columns = set(columns) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        if len(columns) >= len(self._data.columns):
            raise ValueError("Cannot drop all columns from a dataset")

        return Dataset(self._data.drop(columns))

    def rename_columns(self, mapping: dict[str, str]) -> "Dataset":
        missing_columns = set(mapping) - set(self._data.columns)

        if missing_columns:
            raise ValueError(f"Columns not found: {sorted(missing_columns)}")

        if not mapping:
            return Dataset(self._data.clone())

        if any(not isinstance(name, str) or not name for name in mapping.values()):
            raise ValueError("New column names must be non-empty strings")

        resulting_names = [mapping.get(column, column) for column in self._data.columns]

        if len(resulting_names) != len(set(resulting_names)):
            raise ValueError("Renaming would create duplicate column names")

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
        if (
            isinstance(train_ratio, bool)
            or not isinstance(train_ratio, (int, float))
            or not math.isfinite(train_ratio)
            or not 0 < train_ratio < 1
        ):
            raise ValueError("train_ratio must be a finite number between 0 and 1")

        row_count = self.row_count

        if row_count == 0:
            return Dataset(self._data.clone()), Dataset(self._data.clone())

        if row_count == 1:
            raise ValueError("At least 2 rows are required to split a dataset")

        train_count = math.floor(row_count * train_ratio)
        train_count = max(1, min(train_count, row_count - 1))

        indices = list(range(row_count))
        random.Random(seed).shuffle(indices)

        train_indices = indices[:train_count]
        test_indices = indices[train_count:]

        return (
            Dataset(self._data[train_indices]),
            Dataset(self._data[test_indices]),
        )

    def stratified_split(
        self,
        column: str,
        train_ratio: float = 0.8,
        seed: int | None = None,
    ) -> tuple["Dataset", "Dataset"]:
        if (
            isinstance(train_ratio, bool)
            or not isinstance(train_ratio, (int, float))
            or not math.isfinite(train_ratio)
            or not 0 < train_ratio < 1
        ):
            raise ValueError("train_ratio must be a finite number between 0 and 1")

        if column not in self._data.columns:
            raise ValueError(f"Column not found: {column}")

        groups = defaultdict(list)

        for index, value in enumerate(self._data[column].to_list()):
            if value is None:
                raise ValueError(f"Column '{column}' contains null values")

            if isinstance(value, float) and math.isnan(value):
                raise ValueError(f"Column '{column}' contains NaN values")

            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    f"Column '{column}' must contain hashable scalar values "
                    "for stratified splitting"
                ) from error

            groups[value].append(index)

        rng = random.Random(seed)
        train_indices = []
        test_indices = []

        for value, indices in groups.items():
            if len(indices) < 2:
                raise ValueError(
                    f"Every class must have at least 2 samples. "
                    f"Class '{value}' has {len(indices)}."
                )

            rng.shuffle(indices)

            train_count = math.floor(len(indices) * train_ratio)
            train_count = max(1, min(train_count, len(indices) - 1))

            train_indices.extend(indices[:train_count])
            test_indices.extend(indices[train_count:])

        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        return (
            Dataset(self._data[train_indices]),
            Dataset(self._data[test_indices]),
        )

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
