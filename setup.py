from setuptools import setup, find_packages
setup(
    name="dstrf",
    description="MEG/EEG analysis tools",
    version="0.1",
    packages=find_packages(),
    python_requires='>=2.7, <3.0',

    install_requires=[
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
