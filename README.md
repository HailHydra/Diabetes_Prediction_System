
Diabetes Prediction System
==========================

The Diabetes Prediction System leverages machine learning to predict the likelihood of diabetes based on user-provided medical and demographic data. This web application aims to support early detection and proactive diabetes management through real-time predictions.

Features
--------
- User-Friendly Web Interface: Easily accessible web application for inputting medical data.
- Intel Optimized Libraries: Efficient data processing using scikit-learn-intelex and modin.pandas.
- Predictive Analysis: Robust diabetes prediction powered by the XGBoost model.

Setup Instructions
------------------

Prerequisites
-------------
- Python (version 3.7 or higher) installed on your system.
- Git Bash for initializing Ray locally.
- Recommended: A Python environment manager such as Conda.

Step 1: Set Up the Python Environment
-------------------------------------

1. Create a virtual environment:
    - For venv:
        python -m venv env
    - For Conda:
        conda create --name diabetes-prediction python=3.12.7

2. Activate the environment:
    - For venv:
        - On macOS/Linux:
            source env/bin/activate
        - On Windows:
            env\Scripts\activate
    - For Conda:
        conda activate diabetes-prediction

Step 2: Install Required Packages
---------------------------------
Run the following command to install all dependencies:
    pip install -r requirements.txt

If you are using anaconda for environment management, install sklearnex using conda-forge:
    conda install -c conda-forge scikit-learn-intelex

Step 3: Install the Package
---------------------------
Run the following command to install the package using setup.py:
    python setup.py install

Step 4: Initialize Ray for modin.pandas
-----------------------------------------
To enable parallelized data processing, start Ray locally using Git Bash:
    ray start --head

Step 5: Launch the Application
------------------------------
Run the application script:
    python app.py

Step 6: Access the Web Application
----------------------------------
Open your browser and navigate to:
    http://localhost:8000

How to Use
----------
1. Enter the requested medical details into the input form provided on the web interface.
2. Submit the form to receive an instant diabetes risk assessment.

Technical Details
-----------------
Tools & Libraries
- Programming Language: Python
- Web Framework: Flask
- Libraries: Intel-optimized Scikit-learn, XGBoost, modin.pandas
- Dataset: Pima Indians Diabetes Database

Intel Technologies
- scikit-learn-intelex: Accelerated model training.
- modin.pandas: Efficient data preprocessing.
- Intel-optimized XGBoost: Enhanced model inference.

Future Scope
------------
- Cloud deployment for scalability.
- Incorporation of SHAP for explainable AI insights.
- Integration with hospital management systems.

Acknowledgments
---------------
This project was developed for the Intel AI Hackathon @ IEEE INDICON 2024 by The Mavericks from SASTRA Deemed University.

Team Members:
- Varun M
- Suryaprakas B A

For queries, contact: varun.m.karur@gmail.com