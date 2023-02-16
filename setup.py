
# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/

from setuptools import setup

setup(
    name='brat_scoring',
    version='0.2.0',
    description='Python package for comparing BRAT annotations',
    url='https://github.com/Lybarger/brat_scoring.git',
    author='Kevin Lybarger',
    author_email='klybarge@gmu.edu',
    license='MIT License',
    packages=['brat_scoring'],
    install_requires=['wheel',
                      'pandas',
                      'tqdm',
                      'numpy',
                      'spacy>=3.0.0',
                      ],

    classifiers=[
        'Development Status :: Updated since n2c2 challenge for error handling',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
