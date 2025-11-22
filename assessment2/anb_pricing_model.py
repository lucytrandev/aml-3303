import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import boto3
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ==========================================
# 1. Data Loading (AWS S3 Version)
# ==========================================
def load_data_from_s3(bucket_name, file_key):
    """
    Loads data from AWS S3 using boto3.
    """
    print(f"Connecting to S3 bucket: {bucket_name}...")

    try:
        # Initialize S3 client (uses your local AWS credentials)
        s3 = boto3.client('s3')

        # Get the object
        response = s3.get_object(Bucket=bucket_name, Key=file_key)

        # Read content
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 200:
            print(f"Successfully retrieved {file_key} from S3.")
            csv_content = response['Body'].read()
            df = pd.read_csv(io.BytesIO(csv_content))
            print(f"Data loaded. Shape: {df.shape}")
            return df
        else:
            print(f"Error: S3 returned status code {status}")
            return None

    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return None


# ==========================================
# 2. Data Preprocessing
# ==========================================
def preprocess_data(df):
    """
    Cleans the dataframe and separates features (X) and target (y).
    """
    print("Preprocessing data...")

    # Drop columns that are IDs, names, or text (for this V1 model)
    # 'neighbourhood' is dropped for simplicity in V1 (high cardinality),
    # but you could add it back with proper encoding.
    drop_cols = ['id', 'name', 'host_id', 'host_name', 'last_review', 'neighbourhood']
    df_clean = df.drop(columns=drop_cols, errors='ignore')

    # Handle Missing Values
    # reviews_per_month: NaN usually means 0 reviews
    df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

    # Drop rows where price is 0 (outliers/errors) or missing
    df_clean = df_clean[df_clean['price'] > 0]
    df_clean = df_clean.dropna(subset=['price'])

    # Define Features and Target
    target = 'price'
    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    # Identify numeric and categorical columns
    numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
                        'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    categorical_features = ['neighbourhood_group', 'room_type']

    return X, y, numeric_features, categorical_features


# ==========================================
# 3. Model Training & MLflow Tracking
# ==========================================
def train_model(X, y, numeric_features, categorical_features):
    """
    Builds a pipeline, trains the model, and logs to MLflow.
    """
    # Define Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define Model (Random Forest)
    params = {
        "n_estimators": 100,
        "max_depth": 15,
        "random_state": 42
    }

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(**params))])

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow Run
    # Changed experiment name to indicate S3 source
    mlflow.set_experiment("Airbnb_Price_Prediction_S3")

    with mlflow.start_run():
        print("Training model...")
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Model Metrics -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        # Log Parameters
        mlflow.log_params(params)

        # Log Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log Model
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Run logged to MLflow (Experiment: Airbnb_Price_Prediction_S3).")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    BUCKET_NAME = "aml-3303"
    FILE_KEY = "airbnb/raw_data/listings.csv"

    try:
        # 1. Load from S3
        df = load_data_from_s3(BUCKET_NAME, FILE_KEY)

        if df is not None:
            # 2. Preprocess
            X, y, num_feats, cat_feats = preprocess_data(df)

            # 3. Train & Track
            train_model(X, y, num_feats, cat_feats)

    except Exception as e:
        print(f"An error occurred: {e}")