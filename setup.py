from setuptools import find_packages, setup

setup(
    name="chia",
    version="1.0.0",
    packages=find_packages(),
    python_requires="==3.7",
    install_requires=[
        "pillow~=6.2",
        "tensorflow~=2.0",
        "tensorflow-addons~=0.6",
        "networkx~=2.3",
        "gputil~=1.4",
        "scikit-learn~=0.21",
        "scikit-image~=0.15",
        "matplotlib~=3.1",
        "imageio~=2.6",
        "numpy~=1.17",
        "scipy~=1.3",
        "nltk~=3.4",
        "pyqt5==5.9",
    ],
    # metadata to display on PyPI
    author="Clemens-Alexander Brust",
    author_email="clemens-alexander.brust@uni-jena.de",
    description="Concept Hierarchies for Incremental and Active Learning",
)
