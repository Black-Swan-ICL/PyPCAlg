import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setup(
    name='PyPCAlg',
    version='1.0.2',
    description='A Python implementation of the original PC algorithm.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/Black-Swan-ICL/PyPCAlg',
    author='K. M-H',
    author_email='kmh.pro@protonmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
    ],
    packages=find_packages(),
    install_requires=[
        'numpy>=1.22.0',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'pingouin'
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={'': ['examples/true_independence_relationships_graph_*.csv']}
)
