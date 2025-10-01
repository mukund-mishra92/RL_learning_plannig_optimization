# 📦 Required Dependencies
"""
Install all required packages for the RL Path Planning System
"""

import subprocess
import sys

# Add PyTorch for GPU support
def install_pytorch():
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu118"
    ])

if __name__ == "__main__":
    install_pytorch()

def install_requirements():
    """Install all required packages"""
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "mysql-connector-python>=8.0.0",
        "gym>=0.21.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0"
    ]
    
    print("🔧 Installing required packages...")
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("🎉 All packages installation completed!")

if __name__ == "__main__":
    install_requirements()