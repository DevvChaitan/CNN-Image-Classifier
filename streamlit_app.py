"""Streamlit Cloud Default Entrypoint for CIFAR-10 CNN Classifier.

Aliases app.py and CIFAR10_CNN_Streamlit_Project.py to support default Streamlit Cloud deployments.
"""

from CIFAR10_CNN_Streamlit_Project import streamlit_app

if __name__ == "__main__":
    streamlit_app()
