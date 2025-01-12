import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.callbacks import History
import numpy as np

# Load the data
input_pickle_file = "cleaned_data.pkl"
data = pd.read_pickle(input_pickle_file)

# Define input features and target variables
columns_to_include = [
    "Age", "AnnualIncome", "CreditScore", "EmploymentStatus", "EducationLevel", "Experience",
    "LoanAmount", "LoanDuration", "MaritalStatus", "NumberOfDependents", "HomeOwnershipStatus", 
    "MonthlyDebtPayments", "CreditCardUtilizationRate", "NumberOfOpenCreditLines", 
    "NumberOfCreditInquiries", "DebtToIncomeRatio", "BankruptcyHistory", "LoanPurpose", 
    "PreviousLoanDefaults", "SavingsAccountBalance", "CheckingAccountBalance", "TotalAssets", 
    "TotalLiabilities", "MonthlyIncome", "NetWorth", "BaseInterestRate",
    "TotalDebtToIncomeRatio", "LoanApproved", "RiskScore"
]

# Preprocessing categorical columns
categorical_columns = ["EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose"]

def encode_categorical_data(df):
    df = df.copy()
    
    # One-hot encode non-ordinal categorical columns
    df = pd.get_dummies(df, columns=[col for col in categorical_columns], drop_first=True)
    return df

data = encode_categorical_data(data)
# Define input features and target variables

X = data.drop(columns=["BaseInterestRate", "LoanApproved"])
y_interest_rate = data["BaseInterestRate"]  # Target for interest rate
y_loan_percentage = data["LoanApproved"]  # Target for loan percentage

# Split the data into training, validation, and testing sets
X_train, X_temp, y_interest_train, y_interest_temp = train_test_split(
    X, y_interest_rate, test_size=0.25, random_state=42
)
X_val, X_test, y_interest_val, y_interest_test = train_test_split(
    X_temp, y_interest_temp, test_size=0.6, random_state=42
)

_, _, y_loan_train, y_loan_temp = train_test_split(
    X, y_loan_percentage, test_size=0.25, random_state=42
)
_, _, y_loan_val, y_loan_test = train_test_split(
    X_temp, y_loan_temp, test_size=0.6, random_state=42
)

# Define the neural network model
def create_interest_model():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="relu")  # Output layer for [0, 1] range
    ])
    return model

def create_loan_model():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid")
    ])
    return model

# Compile and train the model for interest rate prediction
interest_model = create_interest_model()
interest_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])


# Create a history callback for tracking
history_interest = History()

interest_model.fit(X_train, y_interest_train, validation_data=(X_val, y_interest_val), epochs=100, batch_size=32, verbose=1, callbacks=[history_interest])

# Create a history callback for tracking
history_loan = History()

# Compile and train the model for loan percentage prediction
loan_model = create_loan_model()
loan_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
loan_model.fit(X_train, y_loan_train, validation_data=(X_val, y_loan_val), epochs=100, batch_size=32, verbose=1, callbacks=[history_loan])

# Save the trained models
interest_model.save("interest_rate_model.keras")
loan_model.save("loan_percentage_model.keras")

# Print history metrics
# print("Interest rate model training history:", history_interest.history)
# print("Loan percentage model training history:", history_loan.history)

# Define a custom accuracy metric
def calculate_custom_accuracy(y_true, y_pred, tolerance=0.1):
    """
    Calculate the percentage of predictions within a tolerance of the true values.
    Tolerance is defined as a fraction of the true value (e.g., 10%).
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        tolerance (float): Acceptable percentage deviation (e.g., 0.1 for 10%).

    Returns:
        float: Custom accuracy percentage.
    """
    accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance * y_true)
    return accuracy * 100  # Return as a percentage

# Evaluate both models on the validation dataset
y_interest_pred = interest_model.predict(X_val)
y_loan_pred = loan_model.predict(X_val)

# Calculate custom accuracies
interest_accuracy = calculate_custom_accuracy(y_interest_val, y_interest_pred.flatten(), tolerance=0.05)
_, train_binary_accuracy = loan_model.evaluate(X_train, y_loan_train, verbose=1)

# Print results
print(f"Interest Rate Model - Custom Accuracy (within 5% tolerance): {interest_accuracy:.2f}%")
print(f"Loan Model - Training Binary Accuracy: {train_binary_accuracy * 100:.2f}%")

print("Models have been trained and saved successfully!")