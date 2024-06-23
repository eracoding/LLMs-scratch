import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="minigrad",
    version="0.1.0",
    author="Ulugbek Shernazarov",
    author_email="u.shernaz4rov@gmail.com",
    description="Scalar-valued autograd engine inspired by a small PyTorch-like neural network library. (credits for Andrej Karpathy)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eracoding/LLMs-scratch/minigrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
