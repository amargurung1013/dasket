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
