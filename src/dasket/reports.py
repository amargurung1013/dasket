from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationReport:
    total_count: int
    valid_count: int
    invalid_count: int
    valid_records: list[dict[str, Any]]
    invalid_records: list[dict[str, Any]]

    @property
    def is_valid(self) -> bool:
        return self.invalid_count == 0
