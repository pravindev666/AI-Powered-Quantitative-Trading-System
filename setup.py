from setuptools import setup, find_packages

setup(
    name="niftypred",
    version="4.5.0",
    packages=find_packages() + ['niftypred.models'],
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.2",
        "yfinance>=0.1.70",
        "torch>=2.2.0",
        "safetensors>=0.6.2",
        "pytorch-forecasting>=1.0.0",
        "lightning>=2.0.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
        "python-dateutil>=2.8.2",
        "streamlit>=1.0.0",
        "transformers>=4.57.0",
    ],
    python_requires=">=3.8",
)