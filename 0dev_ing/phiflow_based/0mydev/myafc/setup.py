from setuptools import setup, find_packages

setup(
    name="myafc",
    version="0.0.1",
    description="Local myafc package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
