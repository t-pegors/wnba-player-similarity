import yaml
from pathlib import Path
from unidecode import unidecode


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the project config as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_name(name: str) -> str:
    """
    Normalize a player name for reliable cross-source joining.
    Removes accents, lowercases, and strips extra whitespace.
    e.g. "Satou Sabally" -> "satou sabally"
         "Jonquel Jones" -> "jonquel jones"
    """
    return unidecode(name).lower().strip()


def ensure_dirs(config: dict) -> None:
    """Create raw and processed data directories if they don't exist."""
    Path(config["data"]["raw_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["data"]["processed_path"]).parent.mkdir(parents=True, exist_ok=True)
