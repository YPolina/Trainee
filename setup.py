import setuptools
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="future_sales_prediction_2024",
    version="0.2.3",
    description="A package for feature extraction, hyperopt, and validation schemas",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    author = 'Polina Yatsko',
    author_email="yatsko_polina1@mail.ru",
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis',
        "Operating System :: OS Independent"],
    python_requires='>=3.7,<3.13',
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "matplotlib",
        "seaborn",
        "hyperopt",
        "shap",
        "matplotlib",
        "seaborn",
        "dvc",
        "dvc-gs",
        "google-auth",
        "google-auth-oauthlib",
        "google-cloud-storage"

    ],
    packages=find_packages(),
    include_package_data = True,
    keywords="machine-learning xgboost hyperopt data-science regression",
)
