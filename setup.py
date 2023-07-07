import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='yangDL',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/m1dsolo/yangDL',
    license='MIT',
    author='m1dsolo',
    author_email='yx1053532442@gmail.com',
    description='yangDL',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
