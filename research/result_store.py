"""JSON persistence for research sessions with resume support."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from research.experiments.base import ExperimentResult


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

logger = logging.getLogger(__name__)


class ResultStore:
    """Persist experiment results as JSON, one file per experiment, with a session manifest."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.session_dir / "manifest.json"
        self.manifest: Dict = {}
        self._load_or_create_manifest()

    def _load_or_create_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)
            logger.info(f"Resumed session with {len(self.manifest.get('experiments', []))} prior experiments")
        else:
            self.manifest = {
                "session_id": self.session_dir.name,
                "started_at": datetime.now().isoformat(),
                "experiments": [],
                "status": "running",
            }
            self._save_manifest()

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, cls=_NumpyEncoder)

    def save_result(self, result: ExperimentResult, index: int):
        """Save a single experiment result and update the manifest."""
        filename = f"exp_{index:03d}_{result.experiment_type}.json"
        filepath = self.session_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, cls=_NumpyEncoder)

        self.manifest["experiments"].append({
            "index": index,
            "filename": filename,
            "experiment_type": result.experiment_type,
            "experiment_id": result.experiment_id,
            "strategy": result.strategy,
            "is_significant": result.is_significant,
            "parent_id": result.parent_id,
        })
        self._save_manifest()
        logger.info(f"Saved experiment #{index}: {result.experiment_type} -> {filename}")

    def load_all_results(self) -> List[ExperimentResult]:
        """Load all experiment results from this session."""
        results = []
        for entry in self.manifest.get("experiments", []):
            filepath = self.session_dir / entry["filename"]
            if filepath.exists():
                with open(filepath, "r") as f:
                    data = json.load(f)
                results.append(ExperimentResult(**data))
        return results

    def get_completed_count(self) -> int:
        return len(self.manifest.get("experiments", []))

    def get_completed_types(self) -> List[str]:
        """Return list of (type, strategy, params_hash) for dedup."""
        return [
            (e["experiment_type"], e["strategy"])
            for e in self.manifest.get("experiments", [])
        ]

    def mark_complete(self, stats: Dict = None):
        self.manifest["status"] = "completed"
        self.manifest["completed_at"] = datetime.now().isoformat()
        if stats:
            self.manifest["session_stats"] = stats
        self._save_manifest()

    def mark_failed(self, error: str):
        self.manifest["status"] = "failed"
        self.manifest["error"] = error
        self.manifest["failed_at"] = datetime.now().isoformat()
        self._save_manifest()


def get_session_dir(base_dir: Path, session_name: str = None) -> Path:
    """Create a session directory with timestamp name."""
    if session_name is None:
        session_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return base_dir / session_name
