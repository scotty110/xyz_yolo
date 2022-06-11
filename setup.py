# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="yolo5",  # Required
    version="0.0.1",  # Required
    description="An implementation to serve a yolo model",  # Optional
    #packages=find_packages(where="src"),  # Required
    packages=find_packages(),  # Required
    
    python_requires=">=3.5",
)
