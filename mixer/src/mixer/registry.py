"""Parser for the dataset log (`repo/prototype/sources.log`).

One `|`-separated line per dataset after a header:
`name | load url | local path | date | license`.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    path: Path
    date: str
    license: str

    @property
    def non_commercial(self) -> bool:
        text = self.license.upper()
        return "NON COMMERCIAL" in text or "-NC" in text


def load_registry(log_path: str | Path) -> dict[str, Dataset]:
    lines = Path(log_path).read_text(encoding="utf-8").splitlines()[1:]
    out = {}
    for line in lines:
        if not line.strip():
            continue
        name, url, path, date, license_ = (f.strip() for f in line.split("|", 4))
        out[name.lower()] = Dataset(name, url, Path(path), date, license_)
    return out
