import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mural-mgerasimiuk",
    version="0.0.1",
    author="Michal Gerasimiuk, Dennis Shung",
    author_email="michal.gerasimiuk@yale.edu, dennis.shung@yale.edu",
    description="Random forest for manifold learning with missing values",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgerasimiuk/mural",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: TBD",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)