from setuptools import setup, find_packages

setup(
    name='POSample',
    version='0.2',
    packages=find_packages(),
    description='Python Package for Potential Outcomes and Conditional Probabilities for the project of Social_Mobility_and_Efficiency',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Hongyu Mou',
    author_email='hongyumou@g.ucla.edu',
    url='https://github.com/HongyuMou/POSample',
    license='MIT',
    install_requires=[
        'numpy==1.22.4',
        'pandas==1.3.4',
        'seaborn==0.11.2',
        'matplotlib==3.4.3',
        'scipy==1.10.1',
        'scikit-learn==0.24.2',
        'statsmodels==0.13.5'
    ],
)

