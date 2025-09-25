# Breast Cancer Detection using Machine Learning 🩺

This project focuses on the detection of breast cancer using various classification algorithms. The goal is to build and evaluate models that can accurately predict whether a tumor is malignant or benign based on several key features.

The analysis and modeling are performed on the well-known **Wisconsin Breast Cancer dataset** from Kaggle.

-----

## 📖 Dataset

The dataset used for this project is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). It contains 569 instances and 32 attributes. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

**Key Features:**

  * `id`: ID number
  * `diagnosis`: The diagnosis of breast tissues (M = malignant, B = benign)
  * `radius_mean`, `texture_mean`, `perimeter_mean`, etc.

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
├── Breast_Cancer_Detection.ipynb   # Main Jupyter Notebook with analysis and modeling
|
├── data.csv/                       # The dataset file
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
    cd Breast\ Cancer\ Detection/
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

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | :------: | :-------: | :----: | :------: |
| Logistic Regression |   0.9737   |   0.9762    |  0.9535  |   0.9647   |
| K-Nearest Neighbors |   0.9474   |   0.9302    |  0.9302  |   0.9302   |
| Support Vector Machine|   0.9825  |   1.0000    |  0.9535  |   0.9762   |
| Decision Tree       |   0.9298   |   0.9070    |  0.9070  |   0.9070   |
| Random Forest       |   0.9561   |   0.9524    |  0.9302  |   0.9412   |


With an accuracy of **98.25%**, the **Support Vector Machine (SVM)** proved to be the most effective and robust model for this classification task.

-----

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the issues page if you want to contribute.

-----