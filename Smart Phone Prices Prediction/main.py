import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Or LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


#load data
df=pd.read_csv('train.csv')
df['price'] = df['price'].map({'not expensive': 0, 'expensive': 1})


# Drop 'price' as it's now the target source
X = df.drop(['price', 'expensive'], axis=1)
y = df['expensive']

# Identify column types
numerical_cols = ['rating','Processor_Series', 'Core_Count', 'Clock_Speed_GHz', 'RAM Size GB', 'Storage Size GB',
                  'battery_capacity', 'fast_charging_power', 'Screen_Size', 'Resolution_Width',
                  'Resolution_Height', 'Refresh_Rate', 'primary_rear_camera_mp', 'num_rear_cameras',
                  'primary_front_camera_mp', 'num_front_cameras', 'memory_card_size']
binary_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster', 'memory_card_support'] 
categorical_cols = ['Processor_Brand', 'Performance_Tier', 'RAM Tier',
                    'Notch_Type', 'os_name', 'os_version', 'brand']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  
)
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) 
])
# Train
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))