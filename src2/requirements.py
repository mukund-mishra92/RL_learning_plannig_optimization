import subprocess
import sys

def install_requirements():
    packages = [
        "torch",
        "numpy"
    ]
    for pkg in packages:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            pkg
        ])

if __name__ == "__main__":
    install_requirements()
