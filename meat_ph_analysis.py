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
   # Ensures analysis isn't skewed or broken from missing values
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

       # Correlation: pH vs Day for each Meat_Type and Group
       # Loop through each meat type
       for meat in df['Meat_Type'].unique():
           # Loop through each group
           for group in ['Fresh', 'Frozen-Thawed']:
               subset = df [(df['Meat_Type'] == meat) & (df['Group'] == group)]
               # Calculate Pearson correlation between Day and pH
               corr, p_value = pearsonr(subset['Day'], subset['pH'])
               # Print correlation and p-value
               print(f"Correlation (pH vs. Day) for {meat} ({group}): {corr:.2f} (p-vale {p_value:.4f}")

        # T-Test: Compare pH between Fresh and Frozen-Thawed for each Meat_Type
        # Loop through each meat type
       for meat in df['Meat_Type'].unique():
           # Get pH for fresh group
           fresh_ph = df[(df['Meat_Type'] == meat) & (df['Group'] == 'Fresh')]['pH']
           # Get pH for frozen-thawed group
           frozen_ph = df[(df['Meat_Type'] == meat) & (df['Group'] == 'Frazen-Thawed')]['pH']
           # Perform t-test
           t_stat, p_value = ttest_ind(fresh_ph, frozen_ph, equal_var=False)
           # Print t-test results
           print(f"T-test (Fresh vs. Frozen-Thawed) for {meat}: t={t_stat:.2f}, p={p_value:4f}")

        # ANOVA: Test pH differences across Meat_Types within each Group
        # Loop through each group
       for group in ['Fresh', 'Frozen-Thawed']:
           # Filter data for group
           group_df = df[df['Group'] == group]
           # Create a list comprehension that builds a list of pH values for each unique meat type in that group
           # Required for ANOVA function bc it needs 3 groups of data to compare
           meat_phs = [group_df[group_df['Meat_Type'] == meat]['pH'] for meat in group_df['Meat_Type'].unique()]
           # Perform ANOVA
           f_stat, p_value = f_oneway(*meat_phs) #*meat_phs unpacks the list into separate arguments for each meat type
           print(f"ANOVA (pH by Meat Type, {group}): F={f_stat:.2f}, p={p_value:.4f}")
    # Handle error during analysis
   except Exception as e:
       print(f"Error analyzing data: {e}")


# Takes the Pandas DataFrame(df)
def plot_data(df):
   """Gnerate plots for pH trends and distributions."""
   # exits the function early of the data is invalid
   if df is None:
       return
   try:
       # Plot 1: pH Over Time by Meat_Type and Group
       plt.figure(figsize=(12, 6))
       sns.lineplot(data=df, x='Date', y='pH', hue='Meat_Type', style='Group', marker='o')
       plt.title('Surface pH of Meat Products Over Time: Fresh vs. Frozen-Thawed', fontsize=14)
       plt.xlabel('Date', fontsize=12)
       plt.ylabel('pH', fontsize=12)
       plt.xticks(rotation=45)
       plt.tight_layout()
       plt.savefig('ph_over_time.png')
       plt.close()

       # Plot 2: Box Plot of pH by eat_Type and Group
       plt.figure(figsize=(10, 6))
       sns.boxplot(data=df, x='Meat_Type', y='pH', hue='Group')
       plt.title('pH Distribution by Meat Type and Storage Condition', fontsize=14)
       plt.xlabel('Meat_Type', fontsize=12)
       plt.ylabel('pH', fontsize=12)
       plt.tight_layout()
       plt.savefig('ph_boxplot.png')
       plt.close()

       # Plot 3: pH vs Day Scatter with Regression Line
       plt.figure(figsize=(10, 6))
       sns.scatterplot(data=df, x='Day', y='pH', hue='Meat_Type', style='Group')
       # Loops over each unique meat type in dataset
       for meat in df['Meat_Type'].unique():
           # for each meat type, loops over both storage conditions
           for group in ['Fresg', 'Frozen-Thawed']:
               # Filters dataframe to only include rows that specific meat + group
               subset = df[(df['Meat_Type'] == meat) & (df['Group'] == group)]
               # uses polyfit to perform linear regression
               slope, intercept = np.polyfit(subset['Day'], subset['pH'], 1)
               plt.plot(subset['Day'], slope * subset['Day'] + intercept, linestyle='--', label=f'{meat} ({group}) Fit')
       plt.title('pH vs. Day with Regression Lines', fontsize=14)
       plt.xlabel('Date', fontsize=12)
       plt.ylabel('pH', fontsize=12)
       plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
       plt.tight_layout()
       plt.savefig('ph_vs_day_scatter.png')
       plt.close()

# defines the main function, entry point of script
def main():
    """Execute data loading, analysis, and plotting."""
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
       print("Analysis complete. Check 'ph_summary_stats.csv', 'ph_over_time.png', 'ph_boxplot.png', and 'ph_vs_day_scatter.png'.")
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
