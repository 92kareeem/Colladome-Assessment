import pandas as pd

def clean_salary_data(csv_path):
#Loads CSV, fills missing salaries with mean, returns average salary after cleaning.
    df = pd.read_csv(csv_path)
    if 'salary' in df.columns:
        df['salary'].fillna(df['salary'].mean(), inplace=True)
        return df['salary'].mean()
    else:
        raise KeyError("The column 'salary' does not exist in the file.")

# usage:
print("Average Salary:", clean_salary_data('salaries.csv'))
