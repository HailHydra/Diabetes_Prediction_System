from setuptools import find_packages,setup

setup(
    name='Diabetes Prediction System',
    version='1.1.1',
    author='Varun',
    author_email='darklususnaturae@gmail.com',
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
        'seaborn',
        'flask',
        'ipykernel',
        'xgboost',
        'requests',
        'aiohttp',
        'scikit-learn-intelex',
        'modin[ray]',
    ],
    packages=find_packages()
)