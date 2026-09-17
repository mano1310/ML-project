from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = ROOT / "artifacts"
    random_state: int = 42
    churn_threshold: float = 0.45


settings = Settings()