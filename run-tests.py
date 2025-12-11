import sys
import subprocess
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.resolve()
    tests_path = repo_root / "tests"
    cmd = [sys.executable, "-m", "pytest", str(tests_path)]
    completed = subprocess.run(cmd)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()

