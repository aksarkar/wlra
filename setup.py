import setuptools

setuptools.setup(
    name='wlra',
    description='Weighted Low Rank Approximation',
    version='0.1',
    url='https://www.github.com/aksarkar/wlra',
    author='Abhishek Sarkar',
    author_email='aksarkar@alum.mit.edu',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    packages=setuptools.find_packages(),
)
