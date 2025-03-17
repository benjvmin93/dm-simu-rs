from setuptools import setup, find_packages
import subprocess
import sys

requirements = []
with open("python/requirements.txt") as f:
    requirements = f.read().splitlines()

# Ensure required dependencies are installed
def check_dependency(cmd, package):
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: {package} is not installed. Please install it first.")
        sys.exit(1)

# Check for Rust compiler and Maturin
check_dependency(["cargo", "--version"], "Cargo")
check_dependency(["maturin", "--version"], "Maturin")

# Run Maturin to build the Rust-based Python module
subprocess.run(["maturin", "develop", "-r"], check=True)

setup(
    name="dm-simu-rs",
    version="0.1.0",
    packages=requirements,
    install_requires=[
        "maturin"
    ],
)
