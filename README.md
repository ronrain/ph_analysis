# Meat pH Analysis Tool

This project analyzes the surface pH and spoilage scores of various meat products over time.

## Features

- Loads and validates meat pH dataset
- Calculates spoilage statistics by meat type
- Computes correlation between pH and spoilage
- Generates plots and saves a PDF report

## Files

- `meat_ph_data.csv` — Sample dataset
- `main.py` — Main script to run analysis
- `summary_stats.csv` — Output stats file
- `meat_ph_analysis_report.pdf` — PDF plot report

## Getting Started

1. Install dependencies:
    ```bash
    pip install pandas seaborn matplotlib scipy
    ```

2. Run the analysis:
    ```bash
    python main.py
    ```

## License

MIT License