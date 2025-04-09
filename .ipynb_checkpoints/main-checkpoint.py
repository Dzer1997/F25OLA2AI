import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



homes= pd.read_csv("DKHousingPricesSample100k.csv")
homes = homes.drop(columns=['city'])
homes['dk_ann_infl_rate%'].fillna(homes['dk_ann_infl_rate%'].median(), inplace=True)
homes['yield_on_mortgage_credit_bonds%'].fillna(homes['yield_on_mortgage_credit_bonds%'].median(), inplace=True)
Q1 = homes['purchase_price'].quantile(0.25)
Q3 = homes['purchase_price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = homes[(homes['purchase_price'] < lower_bound) | (homes['purchase_price'] > upper_bound)]
homes_cleaned = homes[homes['purchase_price'] <= upper_bound]
homes_cleaned.drop(columns=['date', 'quarter', 'address', 'area', 'region','sqm_price'], inplace=True)
homes_cleaned.drop(columns=['house_id'], inplace=True)

X = homes_cleaned.drop("purchase_price", axis=1)
y = homes_cleaned["purchase_price"]

categorical_features = ['house_type', 'sales_type']
numeric_features = [col for col in X.columns if col not in categorical_features]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'house_price_model.pkl')

unique_values1 = homes_cleaned['house_type'].unique()
unique_values2 = homes_cleaned['sales_type'].unique()
unique_values3 = homes_cleaned['year_build'].unique()
print(unique_values1)
print(unique_values3)
print('-----------------------------------------------------')
print(unique_values2)
print('-----------------------------------------------------')
print(unique_values3)


















