Diabetes Prediction System
==========================

This project applies machine learning to predict the likelihood of diabetes based on user-provided medical and demographic data. Designed as a web application, it supports early detection and proactive management of diabetes by generating real-time predictions.

Features
--------
- User-friendly Web Interface: Access predictions by entering basic medical details.
- Intel Optimized Libraries: Includes scikit-learn-intelex and modin.pandas for efficient processing.
- Predictive Analysis: Uses XGBoost for reliable diabetes predictions.

Setup Instructions
------------------

Prerequisites
-------------
- Python installed on your system.
- Git Bash for initializing Ray locally.
- Recommended: Conda or a similar Python environment manager.

Step 1: Set Up the Python Environment
-------------------------------------
1. Create a virtual environment:
   python -m venv env
   or with Conda:
   conda create --name diabetes-prediction python=3.12.7

2. Activate the environment:
   - For venv:
       source env/bin/activate   # On MacOS/Linux
       env\Scripts\activate    # On Windows
   - For Conda:
       conda activate diabetes-prediction

Step 2: Install Required Packages
---------------------------------
Run the following command to install all dependencies:
    pip install -r requirements.txt

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