# Statistic_analysis_API_by_FastAPI

This is a FastAPI based application that allows you to perform various data analysis and visualization tasks. This tool offers a variety of endpoints to perform tasks like generating histograms, bar charts, calculating descriptive statistics, conducting Chi-squared tests, and more. It also includes functions to handle prevalence calculations and logistic regression.

This tool is perfect for users who want to perform data analysis on their datasets and visualize the results quickly and efficiently.

# Features
Perform descriptive statistics on numerical data and generate a histogram.
Analyze categorical data and generate a bar chart.
Calculate Pearson correlation coefficient.
Perform Chi-Squared test for categorical data.
Calculate prevalence of conditions in a dataset.
Perform logistic regression.
Uses OAuth2 for basic token-based authentication.
Usage
This API provides several endpoints for different types of data analysis. All these endpoints accept POST requests with JSON data and return the results in JSON format.

# Here are some of the available endpoints:

/numerical_descriptive: Accepts a list of numerical data and returns a histogram and descriptive statistics.
/categorical_plot: Accepts a list of categorical data and returns a bar chart and a frequency table.
/correlation: Accepts two lists of numerical data and returns the Pearson correlation coefficient.
/chi_squared: Accepts a dictionary of categorical data and performs a Chi-Squared test.
/logistic_regression: Accepts a dataset, dependent variable, and independent variables, performs logistic regression. 

# Installation
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using pip: pip install -r requirements.txt
4. Run the server: uvicorn main:app --reload
5. The server will start on http://localhost:8000. You can interact with it using any HTTP client like curl or Postman.
6. If you need to use python script for request. You need to install the required packages using pip: pip install -r requirment.txt (for request script)

# Contributing
Contributions are welcome. Please submit a pull request or create an issue for any changes or improvements you want to propose.

# License
This project is licensed under the Teerawat Author License. For more information, please contact the author.
