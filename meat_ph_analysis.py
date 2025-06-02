# Import pandas for data manipulation and CSV handling
import pandas as pd
# Import matplotlib.pyplot for creating plots
import matplotlib.pyplot as plt
# Import seaborn for enhanced visualizations
import seaborn as sns
# Import numpy for numerical operations
import numpy as np
# Import pearsonr from scipy.stats for correlation analysis
from scipy.stats import pearsonr, ttest_ind, f_oneway
# Import PdfPages from matplotlib for saving plots to PDF
from matplotlib.backends.backend_pdf import PdfPages

# Set seaborn's white grid style for clean, professional plots
sns.set_style("whitegrid")


# Defines function load_data
# The "file_path='meat_ph_data.csv" sets the argument so if no file is specified, it tries to load this file.
def load_data(file_path='meat_ph_data.csv'):
   """Load and validate pH data from the CSV file."""
   # Try to read the CSV file into a DataFrame
   # "try:" begins a try block - Python will attempt the next lines of code.
   # "pd.read_csv(file_path) uses Pandas to read the CSV into a DataFrame called df"
   try:
       df = pd.read_csv(file_path)
       # Converts the 'Date column from a string (text) to actual datetime objects.
       # Needed for time-based filtering, plotting, and analysis
       df['Date'] = pd.to_datetime(df['Date'])
       # Define required columns for the CSV
       # These are the columns the dataset will be expected to contain
       # Will make sure all the data is here before proceeding
       required_columns = ['Date', 'Meat_Type', 'Group', 'pH']
       # Check if all required columns are present
       # Checks that all required columns exist in the DataFrame
       # If any column is missing, it raises a Value Error that jumps to the except block
       if not all(col in df.columns for col in required_columns):
           raise ValueError(f"CSV must contain columns: {required_columns}")
       # Capitalize Meat_type for consistency
       df['Meat_Type'] = df['Meat_Type'].str.capitalize()
       # Ensure Group is either 'Fresh' or 'Frozen-Thawed'
       df['Group'] = df['Group'].strcapitalize()
       if not all (df['Group'].isin(['Fresh', 'Frozen-thawed'])):
           # Raise error for invalid group values
           raise ValueError("Group column must contain only 'Fresh' or 'Frozen-Thawed")
       # Add 'Day' column (days since first date) for correlation analysis
       df['Day'] = (df['Date'] - df['Date'].min()).dt.days + 1
       # Return the loaded DataFrame once everything checks out and can be used throughout program
       return df
   # Handle case where CSV file is not found
   except FileNotFoundError:
       print("Error: 'meat_ph_data.csv' not found.")
       return None
   # Handle other potential errors
   except Exception as e:
       print(f"Error loading data: {e}")
       return None

# Defines a function named analyze_data
# It takes a DataFrame df as input - returned from the load_data() function
def analyze_data(df):
   """Analyze pH data for fresh vs. frozen-thawed meat, including statistical tests."""
   # Checks if the input data is invalid
   # if it is, the function exits early to avoid errors down the line
   if df is None:
       return
   # Remove any rows that are missing pH or Spoilage_Scores
   # ENsures analysis isn't skewed or broken from missing values
   df = df.dropna(subset=['pH'])
   try:
       # Calculate summary statistics (mean, std, min, max) by Meat_Type and Group
       summary_stats = df.groupby(['Meat_Type', 'Group'])['pH'].describe().round(2)
       # Flatten multi-level columns for clarity
       summary_stats = summary_stats[['mean', 'std', 'min', 'max']]
       # Rename column for readability
       summary_stats.columns = ['pH_Mean', 'pH_Std', 'pH_Min', 'pH_Max']
       # Reset Index to make Meat_Type and Group columns
       summary_stats = summary_stats.reset_index()
       # Save summary statistics to CSV
       summary_stats.to_csv('ph_summary_stats.csv', index=False)
       # Print confirmation
       print("Summary statistics saved to 'ph_summary_stats.csv")
   # Standardizes the text in the Meat_Type column by capitalizing the first letter of each entry
   df['Meat_Type'] = df['Meat_Type'].str.capitalize()
   # Groups the data by Meat_Type
   # Calculates the mean and standard deviation of pH and Spoilage_Score
   # .round(2) makes the output easier to read by rounding to 2 decimal places
   summary_stats = df.groupby('Meat_Type').agg({
       'pH': ['mean', 'std'],
       'Spoilage_Score': ['mean', 'std']
   }).round(2)
   #Flaten multi-level columns to single level
   summary_stats.columns = ['pH_Mean', 'pH_Std', 'Spoilage_Mean', 'Spoilage_Std']
   # Reset index to make Meat_Type a column
   summary_stats = summary_stats.reset_index()
   # Saves the summary statistics to a CSV file
   # No extra headers
   summary_stats.to_csv('summary_stats.csv', index=False)
   # Confirmation message
   print("Summary statistics saved to 'summary_stats.csv")
   # Computes Pearson Correlation between pH and Spoilage_Score
   # corr tells the strength and direction of the relationship
   # p_value tells you if the correlation is statistically significant
   corr, p_value = pearsonr(df['pH'], df['Spoilage_Score'])
   # prints the results of the correlation analysis
   print(f"Correlation between pH and Spoilage Score: {corr:.2f} (p-value: {p_value:.4f})")

# Takes the Pandas DataFrame(df)
def plot_data(df):
   # exits the function early of the data is invalid
   if df is None:
       return
   # Creates a multi-page PDF named meat_ph_analysis_report.pdf and allows you to add multiple plots to it
   with PdfPages('meat_ph_analysis_report.pdf') as pdf:
       # First Plot: pH Over Time
       # Starts a new figure (plot) with width 10 and height 6 inches.
       plt.figure(figsize=(10, 6))
       # Creates a line plot using Seaborn - x-axis is Date, y-axis is pH, color each lone based on Meat_Type and adds markers to each data point
       sns.lineplot(data=df, x='Date', y='pH', hue='Meat_Type', marker='o')
       # Added titles and labels
       plt.title('Surface pH of Meat Products Over Time', fontsize=14)
       plt.xlabel('Date', fontsize=12)
       plt.ylabel('pH', fontsize=12)
       # Rotates x axis so no overlap
       plt.xticks(rotation=45)
       # tight_layput() fixes spacing issues
       plt.tight_layout()
       # Saves plot to PDF
       pdf.savefig()
       #Closes the plot to free memory before generating the next one
       plt.close()


       # Second Plot: Spoilage Score Over Time
       # Starts a new figure (plot) with width 10 and height 6 inches.
       plt.figure(figsize=(10, 6))
       # Creates a line plot using Seaborn - x-axis is Date, y-axis is pH, color each lone based on Meat_Type and adds markers to each data point
       sns.lineplot(data=df, x='Date', y='Spoilage_Score', hue='Meat_Type', marker='o')
       # Added titles and labels
       plt.title('Spoilage Score of Meat Products Over Time', fontsize=14)
       plt.xlabel('Date', fontsize=12)
       plt.ylabel('Spoilage Score (1=Fresh, 5=Spoiled)', fontsize=12)
       # Rotates xaxis so no overlap
       plt.xticks(rotation=45)
       # tight_layput() fixes spacing issues
       plt.tight_layout()
       # Saves plot to PDF
       pdf.savefig()
       #Closes the plot to free memory before generating the next one
       plt.close()

def plot_additional_visuals(df):
    """Generate additional plots for pH and spoilage analysis."""
    try:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Meat_Type', y='pH', data=df)
        plt.title('pH Distribution by Meat Type')
        plt.xlabel('Meat_Type')
        plt.ylabel('pH')
        plt.savefig('ph_boxplot.png')
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='pH', y='Spoilage_Score', hue='Meat_Type', data=df)
        slope, intercept, _, _, _ = stats.linregress(df['pH'], df['Spoilage_Score'])
        plt.plot(df['pH'], slope * df['pH'] + intercept, color='red', linestyle='--')
        plt.title('pH vs Spoilage Score (Correlation: 0.74)')
        plt.xlabel('pH')
        plt.ylabel('Spoilage Score')
        plt.savefig('ph_vs_spoilage_scatter.png')
        plt.close()

    except Exception as e:
        print(f"Error predicting shelf life: {e}")

# Prediction model for Shelf Life
# Using df from DataFrame meat data
# spoilage threshold using 4 as default
def predict_shelf_life(df, spoilage_threshold=4.0):
    """Predict shelf life (days until spoilage) for meat type"""
    try:
        # Filters the DataFrame to only include rows where meat is considereed spoiled
        # Keeps only Meat_type and Date column for analysis
        spoiled = df[df['Spoilage_Score'] >= spoilage_threshold][['Meat_Type', 'Date']]
        # Groups the filtered data by Meat_Type, and finds the earliest spoilage date for each type
        # reset_index() turns the groupby result into a clean DataFrame 
        shelf_life = spoiled.groupby('Meat_Type')['Date'].min().reset_index()
        # sets a fixed date (assumed production or packaging date)
        start_date = pd.to_datetime('2025-06-01')
        # adds a new column and calculates number of days from the start date until spoilage
        shelf_life['Days_Until_Spoilage'] = (shelf_life['Date'] - start_date).dt.days
        print("\nPredicted Shelf Life (days until spoilage score >= 4.0):")
        print(shelf_life[['Meat_Type', 'Days_Until_Spoilage']])
        return shelf_life
    except Exception as e:
        print(f"Error predicting shelf life: {e}")
        return None


# defines the main function, entry point of script
def main():
   # Calls the load_data() function
   # Attempts to read and validate meat_ph_data.csv
   # returns a pandas DataFrame(df) or None if loading fails
    df = load_data()
   # Checks if data was successfully loaded
    if df is not None:
       # Calls analyze_data() function
       # Cleans data, calculates stats by meat_type and prints correlation between pH and spoilage
       analyze_data(df)
       # Calls plot_data(df)
       # Generates two time-series plots (pH and spoilage over time) and saves them to PDF report
       plot_data(df)
       print("Analysis complete. Check 'meat_ph_analysis_report.pdf' and 'summary_stats.csv'. ")
    shelf_life = predict_shelf_life(df)
    # If predicition was successful, saves the shelf life data to a CSV file
    # index=False means dont write the row numbers to the file
    if shelf_life is not None:
        shelf_life.to_csv('shelf_life_prediction.csv', index=False)
   


# Python built-in condition
# __name__ is a special variable that sets
       #if the file is being run directly, __name__ == "__main__"
       #if its being imported as a module in another script, __name__ will equal the name of the file
if __name__ == "__main__":
   # Calls the main() function
   main()
