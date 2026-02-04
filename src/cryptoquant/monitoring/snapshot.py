from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from cryptoquant.config import AppConfig
from cryptoquant.utils.time import utc_now


def write_snapshot(config: AppConfig, payload: Dict[str, Any], name: str) -> Path:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    snapshot = {
        "timestamp": timestamp,
        "repo": config.repo_name,
        "system_goal": config.system_goal,
        "risk": {
            "max_position": config.risk.max_position,
            "max_drawdown": config.risk.max_drawdown,
            "execution_on": config.risk.execution_on,
        },
        "payload": payload,
    }
    path = Path(config.monitoring.log_path) / "snapshots" / f"{name}-{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
    return path
