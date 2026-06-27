import subprocess
import sys


def main():
    a = [
        [sys.executable, "expand_data.py"],
        [sys.executable, "train.py"],
        [sys.executable, "plot.py"],
        [sys.executable, "test.py"],
    ]

    for b in a:
        c = subprocess.run(b, check=False)
        if c.returncode != 0:
            sys.exit(c.returncode)

    print("pipeline done")


if __name__ == "__main__":
    main()
