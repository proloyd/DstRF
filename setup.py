from setuptools import setup, find_packages
setup(
    name="dstrf",
    description="MEG/EEG analysis tools",
    version="0.1",
    packages=find_packages(),
    python_requires='>=3.0',

    install_requires=[
        'numpy',
        'scipy',
        'eelbrain',
        'nilearn',
    ],


    # metadata for upload to PyPI
    author="Proloy DAS",
    author_email="proloy@umd.com",
    license="apache 2.0",
    project_urls = {
        "Source Code": "https://github.com/proloyd/fastapy",
    }
)
