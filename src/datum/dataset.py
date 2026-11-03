from pathlib import Path

import polars as pl


class Dataset:
    def __init__(self, data: pl.DataFrame):
        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be a Polars DataFrame")

        self._data = data

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
