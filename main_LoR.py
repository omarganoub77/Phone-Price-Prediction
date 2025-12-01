import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# load data
df = pd.read_csv("train.csv")

# map target
df["price"] = df["price"].map({"non-expensive": 0, "expensive": 1})

# features/target
X = df.drop(["price"], axis=1)
y = df["price"]

# column groups
numerical_cols = [
    "rating", "Core_Count", "Clock_Speed_GHz", "RAM Size GB",
    "Storage Size GB", "battery_capacity", "fast_charging_power",
    "Screen_Size", "Resolution_Width", "Resolution_Height",
    "Refresh_Rate", "primary_rear_camera_mp", "num_rear_cameras",
    "primary_front_camera_mp", "num_front_cameras"
]

binary_cols = ["Dual_Sim", "4G", "5G", "Vo5G", "NFC", "IR_Blaster", "memory_card_support"]

categorical_cols = [
    "Processor_Brand", "Performance_Tier", "RAM Tier",
    "Notch_Type", "os_name", "os_version", "brand",
    "Processor_Series", "memory_card_size"
]

# simple binary encoding: map Yes/No to 1/0 before pipeline
for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0})

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
)

# split train/validation (learning only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model pipeline with Logistic Regression
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# train
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model
joblib.dump(model, "smartphone_price_logreg_model")
print("Model Saved✅")
