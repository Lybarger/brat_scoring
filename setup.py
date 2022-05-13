
# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/

from setuptools import setup

setup(
    name='brat_scoring',
    version='0.1.7',
    description='Python package for comparing BRAT annotations',
    url='https://github.com/Lybarger/brat_scoring.git',
    author='Kevin Lybarger',
    author_email='lybarger@uw.edu',
    license='MIT License',
    packages=['brat_scoring'],
    install_requires=['wheel',
                      'pandas',
                      'tqdm',
                      'numpy',
                      'spacy>=3.0.0',
                      ],

    classifiers=[
        'Development Status :: Initial release',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
