# Import pandas for data manipulation and CSV handling
import pandas as pd
# Import matplotlib.pyplot for creating plots
import matplotlib.pyplot as plt
# Import seaborn for enhanced visualizations
import seaborn as sns
# Import numpy for numerical operations
import numpy as np
# Import pearsonr from scipy.stats for correlation analysis
from scipy.stats import pearsonr
# Import PdfPages from matplotlib for saving plots to PDF
from matplotlib.backends.backend_pdf import PdfPages

# Set seaborn's white grid style for clean, professional plots
sns.set_style("whitegrid")


# Defines function load_data
# The "file_path='meat_ph_data.csv" sets the argument so if no file is specified, it tries to load this file.
def load_data(file_path='meat_ph_data.csv'):
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
       required_columns = ['Date', 'Meat_Type', 'pH', 'Color', 'Odor', 'Texture', 'Spoilage_Score']
       # Check if all required columns are present
       # Checks that all required columns exist in the DataFrame
       # If any column is missing, it raises a Value Error that jumps to the except block
       if not all(col in df.columns for col in required_columns):
           raise ValueError(f"CSV must contain columns: {required_columns}")
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
