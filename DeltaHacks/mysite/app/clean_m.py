# Import necessary libraries
import pandas as pd

# File path for the CSV file (replace with your file path)
input_csv_file = "Loan.csv"
output_pickle_file = "cleaned_data.pkl"

# List of required columns
required_columns = [
    "Age", "AnnualIncome", "CreditScore", "EmploymentStatus", "EducationLevel", 
    "Experience", "LoanAmount", "LoanDuration", "MaritalStatus", "NumberOfDependents", 
    "HomeOwnershipStatus", "MonthlyDebtPayments", "CreditCardUtilizationRate", 
    "NumberOfOpenCreditLines", "NumberOfCreditInquiries", "DebtToIncomeRatio", 
    "BankruptcyHistory", "LoanPurpose", "PreviousLoanDefaults", "SavingsAccountBalance", 
    "CheckingAccountBalance", "TotalAssets", "TotalLiabilities", "MonthlyIncome", 
    "NetWorth", "BaseInterestRate", "TotalDebtToIncomeRatio", "LoanApproved", "RiskScore"
]

# Load the CSV file into a DataFrame
data = pd.read_csv(input_csv_file)

# Check for missing columns
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing from the input CSV file: {missing_columns}")

# Filter the DataFrame to include only the required columns
data = data[required_columns]

# Drop rows with missing values
data_cleaned = data.dropna()

# Save the cleaned DataFrame as a pickle file
data_cleaned.to_pickle(output_pickle_file)

print(data_cleaned)
print(f"Data cleaning complete. Cleaned data saved as '{output_pickle_file}'")