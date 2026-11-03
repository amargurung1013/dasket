from typing import Any

import polars as pl
from pydantic import BaseModel, ValidationError


def validate_records(
    dataframe: pl.DataFrame,
    schema: type[BaseModel],
) -> dict[str, Any]:
    valid_records = []
    invalid_records = []

    for index, record in enumerate(dataframe.to_dicts()):
        try:
            validated = schema.model_validate(record)
            valid_records.append(validated.model_dump())

        except ValidationError as error:
            errors = [
                {
                    "field": ".".join(str(part) for part in item["loc"]),
                    "error_type": item["type"],
                    "message": item["msg"],
                    "input": item.get("input"),
                }
                for item in error.errors()
            ]

            invalid_records.append(
                {
                    "row": index,
                    "record": record,
                    "errors": errors,
                }
            )

    return {
        "total_count": dataframe.height,
        "valid_count": len(valid_records),
        "invalid_count": len(invalid_records),
        "valid_records": valid_records,
        "invalid_records": invalid_records,
    }
