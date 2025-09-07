import argparse
import joblib
import pandas as pd

model = joblib.load("model\\final_logreg_model.pkl")
scaler = joblib.load("model\\scaler.pkl")

def predict(file_path):
    df = pd.read_csv(file_path)

    required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")
    
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    prediction = model.predict(df[required_cols])
    df["Prediction"] = ["fraud" if p == 1 else "not fraud" for p in prediction]

    return df[["Prediction"]]

def main():
    parser = argparse.ArgumentParser(description="credict card fraud detection")
    parser.add_argument("--input", type=str, required=True, help="path to csv file")
    args = parser.parse_args()

    result = predict(args.input)
    print(result)

if __name__ == "__main__":
    main()
