import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder

# load data
train_df = pd.read_csv("train.csv")

# map target
train_df["price"] = train_df["price"].map({"non-expensive": 0, "expensive": 1})

# features/target
X = train_df.drop(["price"], axis=1)
y = train_df["price"]

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

for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0})

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
         ("bin", OrdinalEncoder(), binary_cols)
    ]
    
)

# split train/validation (learning only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# train
model.fit(X_train, y_train)

"""# evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(model, "smartphone_price_DTree_model")
print("Model Saved✅")"""

# 2. EVALUATE ON test_data.csv
test_df = pd.read_csv("test.csv")
# Assuming test_data.csv has 'price' column
X_test_final = test_df.drop(["price"], axis=1)
y_test_final = test_df["price"].map({"non-expensive": 0, "expensive": 1})

# Apply same preprocessing for binary columns
for col in binary_cols:
    X_test_final[col] = X_test_final[col].map({"Yes": 1, "No": 0})

# Predict
y_pred_final = model.predict(X_test_final)
print("Tree Test Set Accuracy:", accuracy_score(y_test_final, y_pred_final))
print(classification_report(y_test_final, y_pred_final))
joblib.dump(model, "smartphone_price_DTree_model")
print("Model Saved✅")
