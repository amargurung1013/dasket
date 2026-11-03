<div align="center">

<img src="docs/assets/dasket.png" alt="Dasket logo" width="520">

### Prepare and assess datasets for AI workflows.

A composable Python library for dataset ingestion, validation, transformation, profiling, and preparation.

</div>

---

## Overview

**Dasket** is a Python library designed to simplify the process of preparing and assessing datasets before they enter AI and machine learning workflows.

Working with raw data often involves repetitive operations: inspecting schemas, handling missing values, normalizing text, detecting duplicates, validating records, and preparing reproducible dataset partitions.

Dasket provides a consistent interface for these operations while leveraging established data-processing libraries rather than reinventing their underlying functionality.

Built on **Polars**, **Pydantic**, and **PyArrow**, Dasket focuses on composability, predictable behavior, and explicit data transformations.

### Design principles

- **Composability:** Dataset operations return Dataset instances, allowing transformations to be chained or combined.
- **Explicit validation:** Invalid inputs should produce meaningful errors rather than unexpected behavior.
- **Reproducibility:** Randomized dataset splitting supports deterministic seeds.
- **Separation of concerns:** Dasket provides a higher-level dataset interface while delegating dataframe operations to Polars.
- **Minimal abstraction:** Avoid unnecessary frameworks and hide as little of the underlying data-processing behavior as possible.

---

## Features

| Category | Capabilities |
|---|---|
| Data ingestion | CSV, JSON, JSONL, Parquet |
| Data export | CSV, JSON, JSONL, Parquet |
| Schema validation | Pydantic model-based validation |
| Text processing | Whitespace normalization, lowercase conversion |
| Duplicate detection | Duplicate identification by selected columns |
| Missing values | Null-value analysis and row removal |
| Dataset profiling | Dimensions, column types, null counts, duplicate counts |
| Column operations | Selection, removal, renaming |
| Filtering | Polars expression-based filtering |
| Dataset splitting | Random train/test splitting |
| Stratification | Class-preserving train/test splitting |

---

## Architecture

Dasket uses a small wrapper abstraction around Polars DataFrames.

```text
                    Application
                         |
                         v
                 +---------------+
                 |     Dasket     |
                 |               |
                 |    Dataset    |
                 +---------------+
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
        Ingestion   Transformations  Profiling
             |           |           |
             +-----------+-----------+
                         |
                         v
                  Polars DataFrame
                         |
              +----------+----------+
              |                     |
              v                     v
          Validation             Storage
              |                     |
              v                     v
           Pydantic              PyArrow
```

### Technology stack

| Technology | Responsibility |
|---|---|
| Python 3.12+ | Core implementation |
| Polars | Dataframe operations and transformations |
| Pydantic | Schema validation |
| PyArrow | Columnar data interoperability |
| pytest | Automated testing |
| Ruff | Formatting and linting |
| uv | Dependency and environment management |

Dasket deliberately delegates dataframe execution and storage-format handling to established libraries.

---

## Installation

### Requirements

- Python >= 3.12
- uv (recommended)

### Development installation

Clone the repository:

```bash
git clone <repository-url>
cd dasket
```

Install dependencies:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Verify the installation:

```bash
uv run python -c "from dasket import Dataset; print(Dataset)"
```

---

## Quick Start

```python
import polars as pl

from dasket import Dataset

# Create a dataset
df = pl.DataFrame(
    {
        "name": [" Amar ", "SAM", "Alex", "Amar"],
        "age": [21, 24, None, 21],
        "language": ["Python", "Go", "Rust", "Python"],
    }
)

dataset = Dataset.from_dataframe(df)

# Inspect dataset dimensions
print(dataset.shape)
print(dataset.columns)
print(dataset.row_count)

# Normalize text
normalized = dataset.normalize(
    columns=["name", "language"],
    lowercase=["name", "language"],
)

# Inspect missing values
print(normalized.missing_values())

# Remove rows containing null values
cleaned = normalized.drop_missing()

# Generate a dataset profile
profile = cleaned.profile()
print(profile)

# Filter records
filtered = cleaned.filter(pl.col("age") > 20)

# Split into training and testing datasets
train, test = filtered.split(
    train_ratio=0.8,
    seed=42,
)
```

Every transformation returns a new `Dataset` instance. The original dataset remains unchanged.

---

## API Reference

### 1. Dataset Construction

Dasket supports construction from an existing Polars DataFrame or a supported file format.

#### `Dataset.from_dataframe()`

```python
import polars as pl

from dasket import Dataset

df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Amar", "Sam", "Alex"],
    }
)

dataset = Dataset.from_dataframe(df)
```

#### `Dataset.from_file()`

```python
dataset = Dataset.from_file("data.csv")
```

Supported extensions:

| Extension | Polars reader |
|---|---|
| `.csv` | `pl.read_csv` |
| `.json` | `pl.read_json` |
| `.jsonl` | `pl.read_ndjson` |
| `.parquet` | `pl.read_parquet` |

File extensions are matched case-insensitively.

Unsupported formats raise `ValueError`. Missing files and non-file paths raise `FileNotFoundError`.

#### Dataset properties

```python
dataset.data
dataset.shape
dataset.columns
dataset.row_count
```

| Property | Description |
|---|---|
| `data` | Underlying Polars DataFrame |
| `shape` | Tuple containing rows and columns |
| `columns` | List of column names |
| `row_count` | Number of rows |

---

### 2. Schema Validation

Dasket integrates with Pydantic to validate dataset records against Python models.

```python
from pydantic import BaseModel

from dasket import Dataset


class User(BaseModel):
    name: str
    age: int
    email: str


dataset = Dataset.from_file("users.json")

report = dataset.validate(User)

print(report)
```

Validation is delegated to Dasket's validation module, which uses the supplied Pydantic model as the record schema.

This allows application developers to define their own validation requirements rather than relying on a fixed collection of predefined schemas.

---

### 3. Text Normalization

#### `normalize(columns, lowercase=None)`

Normalizes selected string columns by stripping surrounding whitespace and optionally converting values to lowercase.

```python
normalized = dataset.normalize(
    columns=["name", "email"],
    lowercase=["email"],
)
```

Behavior:

- Removes surrounding whitespace.
- Supports optional lowercase conversion.
- Preserves null values.
- Rejects missing columns.
- Rejects non-string columns.
- Requires lowercase columns to be included in the selected columns.
- Does not modify the original dataset.

Example:

```text
Before:

"  AMAR  "
"  Sam  "
"  ALEX  "

After:

"amar"
"Sam"
"ALEX"
```

---

### 4. Duplicate Detection

#### `find_duplicates(columns=None)`

Identifies duplicate records based on selected columns.

```python
duplicates = dataset.find_duplicates()
```

Use specific columns:

```python
duplicates = dataset.find_duplicates(columns=["email"])
```

If no columns are specified, all columns are used.

**Return semantics:** The method returns one representative row for each duplicate group, retaining the first occurrence and preserving group order.

Example:

```text
Original:

id    name
1     Amar
1     Amar
2     Sam
2     Sam
3     Alex

Result:

id    name
1     Amar
2     Sam
```

This method does not return every occurrence belonging to a duplicate group.

---

### 5. Missing-Value Analysis

#### `missing_values()`

Returns the number of null values in each column.

```python
missing = dataset.missing_values()

print(missing)
```

Example:

```python
{
    "name": 0,
    "age": 3,
    "email": 2,
}
```

The operation counts actual null values, not empty strings or whitespace-only strings.

#### `drop_missing(columns=None)`

Removes rows containing null values.

Remove rows with nulls in any column:

```python
cleaned = dataset.drop_missing()
```

Consider only selected columns:

```python
cleaned = dataset.drop_missing(columns=["email"])
```

An empty column selection is treated as a no-op.

---

### 6. Dataset Profiling

#### `profile()`

Generates a structural summary of the dataset.

```python
report = dataset.profile()

print(report)
```

Example structure:

```python
{
    "row_count": 1000,
    "column_count": 3,
    "columns": {
        "name": {
            "dtype": "String",
            "null_count": 5,
        },
        "age": {
            "dtype": "Int64",
            "null_count": 12,
        },
        "email": {
            "dtype": "String",
            "null_count": 3,
        },
    },
    "duplicate_count": 8,
}
```

The profile includes:

- Total row count.
- Total column count.
- Column data types.
- Null count per column.
- Duplicate row count.

**Important:** `duplicate_count` counts all rows belonging to duplicate groups, not the number of duplicate groups or only the additional occurrences.

---

### 7. Column Operations

#### `select_columns(columns)`

Selects specific columns.

```python
subset = dataset.select_columns(["name", "email"])
```

Column order follows the supplied list.

At least one column must be selected.

#### `drop_columns(columns)`

Removes selected columns.

```python
reduced = dataset.drop_columns(["temporary_field", "internal_id"])
```

Dasket rejects attempts to drop all columns.

#### `rename_columns(mapping)`

Renames columns using a mapping.

```python
renamed = dataset.rename_columns(
    {
        "user_name": "name",
        "user_email": "email",
    }
)
```

The operation validates that:

- Original column names exist.
- New names are non-empty strings.
- Renaming does not create duplicate column names.

---

### 8. Expression-Based Filtering

#### `filter(expression)`

Dasket exposes Polars expression-based filtering through the Dataset abstraction.

```python
import polars as pl

adults = dataset.filter(pl.col("age") >= 18)
```

Multiple conditions:

```python
filtered = dataset.filter((pl.col("age") >= 18) & (pl.col("country") == "India"))
```

String filtering:

```python
filtered = dataset.filter(pl.col("email").str.ends_with("@gmail.com"))
```

Dasket accepts Polars expressions rather than implementing a separate filtering language.

Invalid expression types raise `TypeError`. Column resolution and expression execution are handled by Polars.

---

### 9. Dataset Splitting

#### `split(train_ratio=0.8, seed=None)`

Splits a dataset into training and testing partitions.

```python
train, test = dataset.split(
    train_ratio=0.8,
    seed=42,
)
```

The split:

- Randomly shuffles row indices.
- Uses the supplied ratio to determine the training partition size.
- Preserves all rows across both partitions.
- Supports reproducibility through a random seed.
- Does not modify the original dataset.

For datasets containing at least two rows, both partitions contain at least one row.

An empty dataset produces two empty partitions. A one-row dataset raises `ValueError`.

---

### 10. Stratified Dataset Splitting

#### `stratified_split(column, train_ratio=0.8, seed=None)`

Performs a train/test split while preserving class representation across partitions.

```python
train, test = dataset.stratified_split(
    column="label",
    train_ratio=0.8,
    seed=42,
)
```

For example:

```text
Original distribution:

Class A: 100
Class B: 100
Class C: 100

Training distribution:

Class A: 80
Class B: 80
Class C: 80

Testing distribution:

Class A: 20
Class B: 20
Class C: 20
```

The operation:

1. Groups row indices by class label.
2. Shuffles indices within each class.
3. Calculates the training partition size independently for each class.
4. Ensures each class appears in both partitions.
5. Shuffles the resulting training and testing indices.

Constraints:

- The specified column must exist.
- Null class labels are rejected.
- NaN class labels are rejected.
- Class labels must be hashable.
- Every class must contain at least two samples.
- The training ratio must be between zero and one.

The random seed makes the operation reproducible.

---

### 11. Dataset Export

#### `to_file(path)`

Exports the underlying DataFrame to a supported file format.

```python
dataset.to_file("output.csv")
```

Supported formats:

```python
dataset.to_file("output.csv")
dataset.to_file("output.json")
dataset.to_file("output.jsonl")
dataset.to_file("output.parquet")
```

The destination directory is created automatically when necessary.

Unsupported extensions raise `ValueError`.

---

## Testing

Dasket uses pytest for automated testing and Ruff for formatting and linting.

Run the complete test suite:

```bash
uv run pytest -v
```

Run linting:

```bash
uv run ruff check .
```

Format the codebase:

```bash
uv run ruff format .
```

Run the complete project checks:

```bash
make check
```

The `make check` target executes formatting, linting, and automated tests.

---

## Project Structure

```text
dasket/
│
├── src/
│   └── dasket/
│       ├── __init__.py
│       ├── dataset.py
│       ├── reports.py
│       └── validation.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_validation.py
│   ├── test_normalization.py
│   ├── test_duplicates.py
│   ├── test_missing_values.py
│   ├── test_drop_missing.py
│   ├── test_columns.py
│   ├── test_filter.py
│   ├── test_split.py
│   └── test_stratified_split.py
│
├── docs/
│   └── assets/
│       └── dasket.png
│
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Design Scope

Dasket is intended to provide practical dataset preparation and assessment primitives.

It is not intended to replace:

- Polars or Pandas as dataframe engines.
- Pydantic as a validation framework.
- PyArrow as a columnar data interchange layer.
- Dedicated machine learning frameworks.
- Distributed data-processing engines.

Instead, Dasket provides a consistent interface that combines common dataset preparation operations into one composable API.

---

## Roadmap

Potential future directions:

- [ ] Richer dataset profiling and statistical summaries.
- [ ] Configurable missing-value strategies.
- [ ] Dataset comparison and drift analysis.
- [ ] Schema inference.
- [ ] More advanced dataset sampling strategies.
- [ ] Improved validation diagnostics.
- [ ] Dataset lineage and transformation tracking.

These are potential directions, not currently implemented features.

---

## Contributing

Contributions, bug reports, and suggestions are welcome.

For development:

```bash
git clone <repository-url>
cd dasket

uv sync

make check
```

Please ensure that formatting, linting, and tests pass before submitting changes.

---

## License

See the repository's license file for licensing information.