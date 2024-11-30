from setuptools import find_packages,setup

setup(
    name='Intel_Hack',
    version='1.1.1',
    author='HailHydra',
    author_email='darklususnaturae@gmail.com',
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
        'seaborn',
        'flask',
        'ipykernel',
        'xgboost',
        'requests aiohttp',
        'scikit-learn-intelex',
        'modin[ray]'
    ],
    packages=find_packages()
)