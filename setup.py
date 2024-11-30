import setuptools
from setuptools import setup, find_packages
import subprocess


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class PostInstallCommand(install):
    """Post-installation for installation mode"""
    def run(self):
        install.run(self)  # Run the standard install process
        subprocess.call(['python', '-m', 'future_sales_prediction_2024.run_post_install']) 

setuptools.setup(
    name="future_sales_prediction_2024",
    version="2.2.6",
    description="A package for feature extraction, hyperopt, and validation schemas",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    author = 'Polina Yatsko',
    author_email="yatsko_polina1@mail.ru",
    license='MIT',
    classifiers=[
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
        "dvc[gs]",
        "dvc[gcs]",
        "google-auth",
        "google-auth-oauthlib",
        "google-cloud-storage",
        "argparse",
        "gcsfs"
    ],
    packages=find_packages(),
    include_package_data = True,
    cmdclass={
        'install': PostInstallCommand,
    },
    keywords="machine-learning xgboost hyperopt data-science regression",
)
