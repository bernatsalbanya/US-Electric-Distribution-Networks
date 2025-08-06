from setuptools import setup, find_packages

setup(
    name="US-Electric-Distribution-Networks",  # Package name
    version="0.1.0",  # Version number
    #authors="Bernat Salbanya, Jordi Nin and Ramon Gras",  # Authors
    #correspondent_author="bernat.salbanya@esade.edu",
    description="A Python package for preparing datasets for US Electric Distribution Networks analysis.",
    long_description=open("README.md").read(),  # Use the README as a long description
    long_description_content_type="text/markdown",
    url="https://github.com/bernatsalbanya/US-Electric-Distribution-Networks",  # Change to your repository URL
    packages=find_packages(where="src"),  # Include all packages in the 'src' directory
    package_dir={"": "src"},  # Define the package root directory
    install_requires=[
        "Cython",  # Cython is required for some dependencies
        "argparse",  # For command-line argument parsing
        "setuptools",  # For building and distributing the package
        "numpy==2.2.0",
        "pandas",
        "contourpy==1.3.2",
        "scipy==1.15.3",
        "geopandas",
        "shapely",
        "seaborn",
        "matplotlib",
        "networkx",
        "momepy",
    ],  # Dependencies (same as requirements.txt)
    classifiers=[
        "Programming Language :: Python :: 3",
        "MIT AND (Apache-2.0 OR BSD-2-Clause)",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Minimum Python version
)
