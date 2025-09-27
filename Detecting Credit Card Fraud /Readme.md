# Detecting Credit Card Fraud using Machine Learning 💳

This project aims to build and evaluate various machine learning models to detect fraudulent credit card transactions. The primary challenge in this domain is the highly imbalanced nature of the data, where the number of legitimate transactions far outweighs the fraudulent ones. This project explores different classification algorithms and compares their performance using appropriate evaluation metrics to identify the most effective model for this task.

-----

## 📖 Dataset

The dataset used for this project is the "Credit Card Fraud Detection" dataset available on Kaggle.

Source: Kaggle Credit Card Fraud Detection Dataset

Description: The dataset contains transactions made by European cardholders in September 2013. It presents a total of 284,807 transactions, of which only 492 (0.172%) are fraudulent.

**Features:**

* It contains only numerical input variables which are the result of a PCA transformation. Due to confidentiality issues, original features and background information are not provided.

* Features V1, V2, ... V28 are the principal components obtained with PCA.

* The only features which have not been transformed with PCA are Time and Amount.

* The feature Class is the response variable and it takes value 1 in case of fraud and 0 otherwise.

-----

## 🤖 Models Implemented

Several machine learning classification algorithms were implemented and evaluated to find the most effective model for this prediction task. The models used include:

  * **Logistic Regression:** A linear model for binary classification.
  * **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies based on the majority class of its 'k' nearest neighbors.
  * **Support Vector Machine (SVM):** A powerful classifier that finds an optimal hyperplane to separate data points.
  * **Decision Tree:** A tree-like model of decisions and their possible consequences.
  * **Random Forest:** An ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

-----

## 📂 Project Structure

```
├── model.ipynb   # Main Jupyter Notebook with analysis and modeling
|
├── requirements.txt                # Required Python libraries
|
└── README.md                       # You are here!
```

-----

## ⚙️ Installation and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AkibDa/ml_projects.git
    cd Detecting\ Creditt\ Card\ Fraud/
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**

    ```bash
    jupyter model.ipynb
    ```

-----

## 📊 Results

The performance of each model was evaluated using standard classification metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**. A confusion matrix was also generated for each model to visualize its performance.

|-------------------------------------------------------------------------|
|Model Performance Comparison:                                            |
|-------------------------------------------------------------------------|
|Model                     | Accuracy  | Precision  | Recall   | F1-Score |
|--------------------------|-----------|------------|----------|----------|                                                      
|Logistic Regression       | 0.9239    |  0.9663    | 0.8776   | 0.9198   |
|K-Nearest Neighbors       | 0.7259    |  0.7444    | 0.6837   | 0.7128   |
|Support Vector Machine    | 0.5178    |  0.5152    | 0.5204   | 0.5178   |
|Decision Tree             | 0.9086    |  0.9348    | 0.8776   | 0.9053   |
|Random Forest             | 0.9289    |  0.9884    | 0.8673   | 0.9239   |
|-------------------------------------------------------------------------|


With an accuracy of **92.89%**, the **Random Forest** proved to be the most effective and robust model for this classification task.

-----

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the issues page if you want to contribute.

-----