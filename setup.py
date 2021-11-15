import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rvnn",
    version="0.0.1",
    author="Antoine Simoulin",
    author_email="antoine.simoulin@gmail.com",
    description="A PyTorch package for recursive neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntoineSimoulin/pytree/tree/main",
    download_url="https://github.com/AntoineSimoulin/pytree/archive/refs/tags/v0.0.1-alpha.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)