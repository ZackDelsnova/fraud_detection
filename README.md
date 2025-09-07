# Python Fraud Detection ML Project

## Goal

Build a machine learning model that can classify transactions as “fraud” or “not fraud.”

## Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
and place `creditcard.csv` inside the `data/` folder.

## Pretrained Model

After training, the model is saved as `final_logreg_model.pkl` (not included in repo).

## Libraries used

- pandas
- sklearn
- joblib
- matplotlib
- imbalanced-learn (imblearn)
- jupyter notebook

## Use

Run

```bash
    python fraud_detector.py --input <path_to_csv_file>
```
