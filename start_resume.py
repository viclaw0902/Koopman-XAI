from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "check_environment.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-u", str(ROOT / "Original_structure" / "run_experiments.py")], cwd=ROOT / "Original_structure", check=True)
