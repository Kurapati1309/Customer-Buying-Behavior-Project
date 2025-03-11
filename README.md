# Customer-Buying-Behavior-Project
Developed a Random Forest model to predict customer booking completion, including data cleaning, model evaluation and feature importance analysis.

# Customer Buying Behaviour Prediction Project
# Project Overview
This project predicts customer booking completion using a dataset of customer booking records. A Random Forest model is trained on various features (e.g., number of passengers, sales channel, trip type, purchase lead time, etc.) to forecast whether a customer will complete a booking. The project includes data exploration, cleaning, feature engineering, model training, evaluation, and interpretation through feature importance analysis. A PowerPoint slide summarizing key findings is generated as the final deliverable.

# Project Structure

CustomerBuyingBehaviorProject/
├── customer_booking.csv         # Input dataset with customer booking records
├── main.py                      # Python script for data processing, model training, and presentation creation
├── feature_importance.png       # Generated feature importance plot (top 15 features)
├── Customer_Buying_Behaviour_Analysis.pptx  # PowerPoint slide summarizing key findings
└── README.md                    # This file

# Requirements
Python: 3.12 (or compatible version)
Libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn
python-pptx

# Install the required libraries using:
pip install pandas numpy scikit-learn matplotlib seaborn python-pptx
How to Run the Project
Place the Data File:
Ensure that customer_booking.csv is in the root directory of the project.

# Run the Script:
Open a terminal or command prompt in the project folder and execute:
python main.py

# Outputs Generated:
Data Exploration & Cleaning: The script loads and prints dataset information and basic statistics.
Model Training & Evaluation: It splits the data into training and testing sets, trains a Random Forest model, and prints evaluation metrics (classification report, confusion matrix, test accuracy, and cross-validation scores).
Feature Importance Plot: A high-resolution plot of the top 15 most important features is saved as feature_importance.png.
Presentation: A PowerPoint slide summarizing the model’s performance and key variables is generated as Customer_Buying_Behaviour_Analysis.pptx.
Model Evaluation
Test Accuracy: Approximately 86%
Mean Cross-Validation Score: Approximately 73.6%
The feature importance plot visually highlights the most influential features in predicting customer booking completion.
License & Software
License:
This project is licensed under the MIT License.

# Software & Libraries Used:
Python 3.12
pandas
numpy
scikit-learn
matplotlib
seaborn
python-pptx

# Future Enhancements
Explore additional models (e.g., Logistic Regression, Gradient Boosting) to compare performance.
Enhance feature engineering and perform hyperparameter tuning to improve the model.
Incorporate advanced interpretability techniques (such as SHAP) for deeper insights into feature contributions.

# Acknowledgements
This project was developed as part of a Forage Virtual Experience. Special thanks to the resources provided, including sample Jupyter Notebooks and documentation on web scraping, data manipulation, and machine learning in Python.

