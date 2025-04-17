# Spam Mail Prediction Project

## Description

This project aims to build a machine learning model to classify emails as spam or ham (non-spam). By leveraging natural language processing (NLP) techniques, the model extracts features from email messages and uses a classification algorithm to predict whether an email is spam. This automated system addresses the challenge of filtering unwanted and potentially harmful emails, enhancing the efficiency and security of email communication.

## Problem Statement

Spam emails are a persistent issue, causing inconvenience, wasting time, and posing security risks. The objective of this project is to develop a reliable, automated system that accurately identifies and filters spam emails, improving the user experience and safeguarding communication channels.

## Data

The project uses a dataset of 5,572 email messages, each labeled as either "spam" or "ham." The dataset is split into training (80%, 4,457 samples) and test (20%, 1,115 samples) sets. The data is stored in a CSV file named `mail_data.csv`, which is included in the repository.

| Dataset Details | Value |
| --- | --- |
| Total Samples | 5,572 |
| Training Samples | 4,457 (80%) |
| Test Samples | 1,115 (20%) |
| Columns | Category (spam/ham), Message |

## Methodology

The project follows a structured machine learning pipeline to classify emails:

1. **Data Preprocessing**:

   - Loaded the dataset from `mail_data.csv` using Pandas.
   - Replaced missing values with empty strings.
   - Encoded labels: "spam" as 0, "ham" as 1.

2. **Feature Extraction**:

   - Converted email text into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer from scikit-learn.
   - Removed English stop words and converted text to lowercase for consistency.

3. **Model Training**:

   - Trained a Logistic Regression model using scikit-learn's LogisticRegression on the training data.

4. **Model Evaluation**:

   - Evaluated model performance on both training and test sets using accuracy scores.
   - Achieved approximately 96.77% accuracy on the training set and 96.68% on the test set.

5. **Predictive System**:

   - Implemented a system to classify new email messages.
   - Example: The message "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030" was correctly classified as "Spam."

## Results

The Logistic Regression model demonstrated strong performance, with the following accuracy metrics:

| Dataset | Accuracy |
| --- | --- |
| Training Set | 96.77% |
| Test Set | 96.68% |

These results suggest the model effectively distinguishes between spam and ham emails and generalizes well to unseen data. The predictive system successfully classified sample inputs, indicating practical applicability for email filtering.

## Conclusion

This project successfully developed a spam email detection system using machine learning and NLP techniques. The high accuracy achieved highlights the model's potential for real-world email filtering applications. The combination of TF-IDF feature extraction and Logistic Regression provides a robust framework for addressing spam-related challenges in email communication.

## Future Work

To enhance the project, consider the following:

- Experiment with alternative algorithms like Naive Bayes, Support Vector Machines, or Deep Learning models to compare performance.
- Incorporate additional features, such as email headers or sender information, to improve accuracy.
- Test the model on larger, more diverse datasets to ensure robustness across different email types.

## Dependencies

To run the project, ensure the following are installed:

- Python 3.x
- Pandas
- Scikit-learn
- NumPy

You can install dependencies using pip:

```bash
pip install pandas scikit-learn numpy
```

## How to Run

1. Clone this repository to your local machine.
2. Open the notebook `Spam_Mail_Prediction_Project.ipynb` in Jupyter Notebook or Google Colab.
3. Ensure all dependencies are installed.
4. Run the notebook cells in sequence to preprocess data, train the model, and evaluate results.

## Data Source

The dataset is included in the repository as `mail_data.csv`. It contains 5,572 email messages labeled as "spam" or "ham." No external dataset source is specified, but the file is provided for reproducibility.

## Acknowledgments

- The project utilizes tools and libraries from scikit-learn for machine learning and Pandas for data manipulation.
- Inspiration drawn from standard NLP and classification techniques in data science education.
