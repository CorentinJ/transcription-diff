import subprocess
import sys
from pathlib import Path


_ROOT_DIR = Path(__file__).parent.parent
_EXAMPLES_FPATH = _ROOT_DIR / "examples.py"


def test_run_examples():
    """
    Runs the examples script and verifies it does not crash or print anything to stderr.
    """
    result = subprocess.run([sys.executable, _EXAMPLES_FPATH], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
    assert result.stderr == "", result.stdout
