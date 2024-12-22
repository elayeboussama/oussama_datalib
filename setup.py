from setuptools import setup, find_packages

setup(
    name='datalib',
    version='0.1.0',
    description='Simplified data manipulation and analysis tools.',
    author='Oussama ELAYEB',
    author_email='elayeb.oussama2020@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
