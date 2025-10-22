import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the CSV
df = pd.read_csv('soil_samples_large.csv')
X = df.drop('Max Building Height (m)', axis=1)
y = df['Max Building Height (m)']

# Preprocessing
numeric_features = ['Moisture (%)', 'Clay (%)', 'Sand (%)', 'Silt (%)', 'pH', 'Bearing Capacity (kPa)']
categorical_features = ['Soil Type']

numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('encoder', OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'building_height_model.pkl')
