import pandas as pd
import duckdb
import glob
import os

# --------------------------------------------------------------
# Read all CSV files in a specific folder
# --------------------------------------------------------------

# Define the path where your raw data is stored
data_path = "path_to_your_data_folder"  # Replace this with your actual path

# List all CSV files in the folder
csv_files = glob.glob(os.path.join(data_path, "*.csv"))

# Read all CSV files into a list of DataFrames
dfs = [pd.read_csv(file) for file in csv_files]

# Optionally, print the names of the CSV files
print("Files to be loaded:", csv_files)

# --------------------------------------------------------------
# Merge datasets
# --------------------------------------------------------------

# If the datasets need to be merged, ensure they have a common column for merging (e.g., 'id' or 'timestamp')
merged_data = pd.concat(
    dfs, ignore_index=True
)  # Merging all CSVs into a single DataFrame

# You could also merge on specific columns, e.g.:
# merged_data = pd.merge(dfs[0], dfs[1], on='id')

# Print shape and first few rows to verify
print(f"Data shape after merging: {merged_data.shape}")
print(merged_data.head())

# --------------------------------------------------------------
# Working with DuckDB
# --------------------------------------------------------------

# Set up DuckDB in-memory database for efficient querying and handling of large datasets
connection = duckdb.connect()

# If you want to work with your DataFrame in DuckDB, you can register it as a table
connection.register("my_table", merged_data)

# You can now run SQL queries directly on your DataFrame using DuckDB
# Example query: Selecting the first 10 rows
query_result = connection.execute("SELECT * FROM my_table LIMIT 10").fetchdf()

# Display the result
print(query_result)

# --------------------------------------------------------------
# Data Exploration
# --------------------------------------------------------------

# 1. Check for missing values
print("Missing values per column:")
print(merged_data.isnull().sum())

# 2. Summary statistics
print("Summary statistics:")
print(merged_data.describe())

# 3. Data types
print("Data types of columns:")
print(merged_data.dtypes)

# 4. Value counts for categorical columns
for col in merged_data.select_dtypes(include=["object"]).columns:
    print(f"\nValue counts for {col}:")
    print(merged_data[col].value_counts())

# --------------------------------------------------------------
# Working with Datetimes (if applicable)
# --------------------------------------------------------------

# If your dataset includes datetime columns, it's important to parse and format them properly
# Example: Convert a datetime column 'timestamp' into datetime format
if "timestamp" in merged_data.columns:
    merged_data["timestamp"] = pd.to_datetime(merged_data["timestamp"])

# If necessary, create new columns for date and time components
merged_data["year"] = merged_data["timestamp"].dt.year
merged_data["month"] = merged_data["timestamp"].dt.month
merged_data["day"] = merged_data["timestamp"].dt.day
merged_data["hour"] = merged_data["timestamp"].dt.hour
merged_data["minute"] = merged_data["timestamp"].dt.minute

# --------------------------------------------------------------
# Feature Engineering: Extract features from filenames or other columns
# --------------------------------------------------------------


# For example, if filenames contain useful information such as 'category', 'date', or 'sensor'
# You can extract those features for further use
def extract_features_from_filename(file_path):
    file_name = os.path.basename(file_path)
    # Example: Extract category and date from filename
    category = file_name.split("_")[0]
    date = file_name.split("_")[1]
    return category, date


# Apply this to your list of CSVs
file_features = [extract_features_from_filename(file) for file in csv_files]

# --------------------------------------------------------------
# Data Preprocessing
# --------------------------------------------------------------

# 1. Handle missing values
# - Drop rows with missing values
merged_data.dropna(inplace=True)  # Optional: Drop rows with any NaN values
# - Or fill missing values with a placeholder (e.g., mean for numerical columns)
merged_data.fillna(merged_data.mean(), inplace=True)

# 2. Handle outliers
# - For numerical columns, you can use z-scores or IQR to detect and remove outliers
# Example: Remove rows where any numerical value is an outlier
numerical_columns = merged_data.select_dtypes(include=["float64", "int64"]).columns
for col in numerical_columns:
    z_score = (merged_data[col] - merged_data[col].mean()) / merged_data[col].std()
    merged_data = merged_data[
        z_score.abs() <= 3
    ]  # Keep only rows within 3 standard deviations

# 3. Standardize/Normalize features (if necessary for certain models)
# For example, use StandardScaler for normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])

# 4. Encoding categorical variables (if necessary for machine learning)
# For example, use one-hot encoding for categorical columns
merged_data = pd.get_dummies(merged_data, drop_first=True)

# --------------------------------------------------------------
# Export cleaned data
# --------------------------------------------------------------

# Export the cleaned dataset to a new CSV
cleaned_data_path = "path_to_cleaned_data/cleaned_data.csv"
merged_data.to_csv(cleaned_data_path, index=False)

print(f"Cleaned data exported to: {cleaned_data_path}")
