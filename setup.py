from setuptools import setup, find_packages

setup(
    name='alora',
    version='0.1.0',
    description='A short description of alora',
    author='Kristjan Greenewald',
    author_email='kristjan.h.greenewald@ibm.com',
    packages=find_packages(),
    install_requires=[
        "click",  "peft","transformers" # Add any required dependencies here
    ],
    python_requires='>=3.8',
)
