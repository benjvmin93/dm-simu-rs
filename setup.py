from setuptools import setup, find_packages
import subprocess
import sys
import os

# Read dependencies from requirements.txt
with open("python/requirements.txt") as f:
    requirements = f.read().splitlines()

def check_and_install_dependency(cmd, install_cmd, package):
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"{package} is already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{package} not found. Installing...")
        subprocess.run(install_cmd, shell=True, check=True)

# Check and install Cargo if missing
check_and_install_dependency(
    ["cargo", "--version"],
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source $HOME/.cargo/env",
    "Cargo (Rust)"
)

# Check and install Maturin if missing
check_and_install_dependency(
    ["maturin", "--version"],
    sys.executable + " -m pip install maturin",
    "Maturin"
)

# Ensure Cargo's bin directory is in PATH
os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.cargo/bin")

# Run Maturin to build the Rust-based Python module
subprocess.run(["maturin", "develop", "-r"], check=True)

setup(
    name="dm-simu-rs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
)
