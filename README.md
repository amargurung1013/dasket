<div align="center">

<img src="docs/assets/dasket.png" alt="Dasket logo" width="520">

### Prepare and assess datasets for AI workflows.

A composable Python library for dataset ingestion, validation, transformation, profiling, and preparation.

</div>

---
# Dasket

### Dataset preparation and assessment for AI workflows.

Dasket is a composable Python library for preparing, validating, transforming, and assessing datasets before they enter machine learning workflows.

Built on **Polars, Pydantic, and PyArrow**, Dasket provides a consistent interface for common dataset engineering operations without reinventing existing data-processing infrastructure.

- Load datasets from CSV, JSON, JSONL, and Parquet.
- Validate records using Pydantic schemas.
- Normalize text and handle missing values.
- Detect duplicate records.
- Profile datasets and inspect their structure.
- Create reproducible train/test splits.

---

## Installation

Requires Python 3.12 or later.

```bash
pip install dasket
```

Alternatively, using [uv](https://docs.astral.sh/uv/):

```bash
uv add dasket
```

## Quick Start

Create a dataset from a Polars DataFrame and start preparing it.

```python
import polars as pl
from dasket import Dataset

# Create a DataFrame
df = pl.DataFrame({
    "name": [" Amar ", "SAM", "Alex", "Amar"],
    "age": [21, 24, None, 21],
    "email": [
        "AMAR@example.com",
        "sam@example.com",
        "alex@example.com",
        "AMAR@example.com",
    ],
})

# Create a Dataset
dataset = Dataset.from_dataframe(df)

# Inspect the dataset
print(dataset.shape)
print(dataset.profile())

# Normalize text
dataset = dataset.normalize(
    columns=["name", "email"],
    lowercase=["email"],
)

# Inspect missing values
print(dataset.missing_values())

# Remove rows containing null values
dataset = dataset.drop_missing()

# Detect duplicate records
duplicates = dataset.find_duplicates(
    columns=["email"]
)

# Split into training and testing datasets
train, test = dataset.split(
    train_ratio=0.8,
    seed=42,
)

# Export datasets
train.to_file("train.parquet")
test.to_file("test.parquet")
```

Every transformation returns a new `Dataset` instance. The original dataset remains unchanged.

---

## Why Dasket?

Preparing data for machine learning involves more than loading files into a DataFrame.

Developers frequently need to perform repetitive operations before a dataset is ready for a training pipeline:

- Inspecting schemas and column types.
- Identifying missing values.
- Normalizing inconsistent text.
- Detecting duplicate records.
- Validating records against expected schemas.
- Understanding dataset structure.
- Preparing reproducible training and testing partitions.

Dasket brings these operations together through a consistent, composable API.

Rather than implementing its own DataFrame engine, Dasket builds on established libraries and focuses on providing a higher-level dataset interface.

## Features

| Category | Capabilities |
|---|---|
| Data ingestion | CSV, JSON, JSONL, Parquet |
| Data export | CSV, JSON, JSONL, Parquet |
| Schema validation | Pydantic model-based validation |
| Text processing | Whitespace normalization, lowercase conversion |
| Missing values | Null-value analysis and row removal |
| Duplicate detection | Duplicate identification by selected columns |
| Dataset profiling | Dimensions, column types, null counts, duplicate counts |
| Column operations | Selection, removal, renaming |
| Filtering | Polars expression-based filtering |
| Dataset splitting | Random train/test splitting |
| Stratification | Class-preserving train/test splitting |

---

## Core API

### 1. Dataset Construction

Create a dataset from an existing Polars DataFrame:

```python
import polars as pl
from dasket import Dataset

df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Amar", "Sam", "Alex"],
})

dataset = Dataset.from_dataframe(df)
```

Alternatively, load a supported file directly:

```python
dataset = Dataset.from_file("customers.csv")
```

Supported formats:

| Extension | Format |
|---|---|
| `.csv` | CSV |
| `.json` | JSON |
| `.jsonl` | JSON Lines |
| `.parquet` | Parquet |

File extensions are matched case-insensitively.

Unsupported formats raise `ValueError`. Missing files and non-file paths raise `FileNotFoundError`.

### 2. Dataset Properties

Access basic dataset information:

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

### 3. Schema Validation

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

Developers define their own validation requirements through Pydantic models rather than relying on predefined schemas.

### 4. Text Normalization

Normalize selected string columns:

```python
normalized = dataset.normalize(
    columns=["name", "email"],
    lowercase=["email"],
)
```

Behaviour:

- Removes surrounding whitespace.
- Supports optional lowercase conversion.
- Preserves null values.
- Rejects missing columns.
- Rejects non-string columns.
- Requires lowercase columns to be included in the selected columns.
- Does not modify the original dataset.

### 5. Missing-Value Analysis

Inspect the number of null values in each column:

```python
missing = dataset.missing_values()

print(missing)
```

Remove rows containing null values:

```python
# Remove rows with nulls in any column
cleaned = dataset.drop_missing()

# Consider only selected columns
cleaned = dataset.drop_missing(
    columns=["email"]
)
```

An empty column selection is treated as a no-op.

**Note:** Missing-value analysis counts actual null values, not empty strings or whitespace-only strings.

### 6. Duplicate Detection

Identify duplicate records:

```python
duplicates = dataset.find_duplicates()
```

Specify columns to identify duplicates based on selected fields:

```python
duplicates = dataset.find_duplicates(
    columns=["email"]
)
```

If no columns are specified, all columns are used.

The method returns one representative row for each duplicate group, retaining the first occurrence and preserving group order.

It does not return every occurrence belonging to a duplicate group.

### 7. Dataset Profiling

Generate a structural summary:

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

**Note:** `duplicate_count` counts all rows belonging to duplicate groups, not the number of duplicate groups or only the additional occurrences.

### 8. Column Operations

Select specific columns:

```python
subset = dataset.select_columns(
    ["name", "email"]
)
```

Remove columns:

```python
reduced = dataset.drop_columns(
    ["temporary_field", "internal_id"]
)
```

Rename columns:

```python
renamed = dataset.rename_columns({
    "user_name": "name",
    "user_email": "email",
})
```

Dasket validates column names and prevents renaming operations that would create duplicate column names.

Additional constraints:

- Column selection must contain at least one column.
- Dropping all columns is rejected.
- Original column names must exist.
- New names must be non-empty strings.

### 9. Expression-Based Filtering

Dasket exposes Polars expression-based filtering through its Dataset abstraction.

```python
import polars as pl

adults = dataset.filter(
    pl.col("age") >= 18
)
```

Multiple conditions:

```python
filtered = dataset.filter(
    (pl.col("age") >= 18)
    & (pl.col("country") == "India")
)
```

String filtering:

```python
filtered = dataset.filter(
    pl.col("email").str.ends_with("@gmail.com")
)
```

Dasket accepts native Polars expressions rather than implementing a separate filtering language.

Invalid expression types raise `TypeError`. Column resolution and expression execution are handled by Polars.

### 10. Dataset Splitting

Split a dataset into training and testing partitions:

```python
train, test = dataset.split(
    train_ratio=0.8,
    seed=42,
)
```

The operation:

- Randomly shuffles row indices.
- Uses the supplied ratio to determine the training partition size.
- Preserves all rows across both partitions.
- Supports reproducibility through a random seed.
- Does not modify the original dataset.

For datasets containing at least two rows, both partitions contain at least one row.

An empty dataset produces two empty partitions. A one-row dataset raises `ValueError`.

### 11. Stratified Dataset Splitting

Perform a train/test split while preserving class representation:

```python
train, test = dataset.stratified_split(
    column="label",
    train_ratio=0.8,
    seed=42,
)
```

For example, a dataset containing:

```text
Class A: 100
Class B: 100
Class C: 100
```

Can produce:

```text
Training:
Class A: 80
Class B: 80
Class C: 80

Testing:
Class A: 20
Class B: 20
Class C: 20
```

The operation:

- Groups row indices by class label.
- Shuffles indices within each class.
- Calculates the training partition size independently for each class.
- Ensures each class appears in both partitions.
- Shuffles the resulting training and testing indices.

Constraints:

- The specified column must exist.
- Null class labels are rejected.
- NaN class labels are rejected.
- Class labels must be hashable.
- Every class must contain at least two samples.
- The training ratio must be between zero and one.

The random seed makes the operation reproducible.

### 12. Dataset Export

Export a dataset to a supported file format:

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

## Architecture

Dasket uses a small wrapper abstraction around Polars DataFrames.

```text
             Application
                  |
                  v
               Dasket
                  |
          +-------+-------+
          |       |       |
      Ingestion  Transformations  Profiling
          |       |       |
          +-------+-------+
                  |
                  v
            Polars DataFrame
                  |
             +----+----+
             |         |
             v         v
         Validation  Storage
             |         |
             v         v
          Pydantic  PyArrow
```

### Technology Stack

| Technology | Responsibility |
|---|---|
| Python 3.12+ | Core implementation |
| Polars | DataFrame operations and transformations |
| Pydantic | Schema validation |
| PyArrow | Columnar data interoperability |
| pytest | Automated testing |
| Ruff | Formatting and linting |
| uv | Dependency and environment management |

Dasket deliberately delegates DataFrame execution and storage-format handling to established libraries.

---

## Design Scope

Dasket is intended to provide practical dataset preparation and assessment primitives.

It is not intended to replace:

- Polars or Pandas as DataFrame engines.
- Pydantic as a validation framework.
- PyArrow as a columnar data interchange layer.
- Dedicated machine learning frameworks.
- Distributed data-processing engines.

Instead, Dasket provides a consistent interface that combines common dataset preparation operations into one composable API.

## Roadmap

Potential future directions include:

- Richer dataset profiling and statistical summaries.
- Configurable missing-value strategies.
- Dataset comparison and drift analysis.
- Schema inference.
- More advanced dataset sampling strategies.
- Improved validation diagnostics.
- Dataset lineage and transformation tracking.

These are potential directions, not currently implemented features.

---

## Development

Contributions, bug reports, and suggestions are welcome.

Clone the repository:

```bash
git clone https://github.com/amargurung1013/dasket.git
cd dasket
```

Install development dependencies:

```bash
uv sync
```

Run tests:

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

Run all project checks:

```bash
make check
```

The `make check` target executes formatting, linting, and automated tests.

Please ensure that formatting, linting, and tests pass before submitting changes.

## Project Structure

```text
dasket/
├── src/
│   └── dasket/
│       ├── __init__.py
│       ├── dataset.py
│       ├── reports.py
│       └── validation.py
├── tests/
├── docs/
│   └── assets/
│       └── dasket.png
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## License

Dasket is released under the MIT License. See [LICENSE](LICENSE) for details.