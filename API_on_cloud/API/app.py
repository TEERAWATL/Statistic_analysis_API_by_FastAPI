#Importance Library for fastapi
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import secrets
#Importance Library for image creation
import base64
from io import BytesIO
#Importance Library for data analysis
from typing import List, Dict, Union, Any
import json
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pydantic import BaseModel
#Importance Library for logging
import logging

#Config thai font
matplotlib.rc('font', family='tahoma')


#Create api app
app = FastAPI()

#Create token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configure logging
logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)

#Create class for check authenticate
async def verify_token(token: str = Depends(oauth2_scheme)):
    #Set up correct token (You can change it)
    correct_token = "teerawat12345"
    if not secrets.compare_digest(token, correct_token):
        logging.error("Invalid token received.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    #Keep log when token is correct
    logging.info("Token verified successfully.")
    return token

#Create function for generate histogram (returned image)
def generate_histogram(data):
    plt.hist(data, bins=10, edgecolor='black', color='skyblue')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # Convert the plot to an image buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    plt.close()  # Close the plot to free memory
    #Keep log when histogram is generated
    logging.info("Histogram generated successfully.")
    return image_base64

#Create function for generate bar chart (returned image)
def plot_categorical_data(data):
    # Calculate frequency of each category
    df = pd.DataFrame(data, columns=['Values'])
    category_counts = df['Values'].value_counts()

    # Create frequency table and convert to JSON
    freq_table = category_counts.reset_index().to_dict(orient='records')
    freq_table_json = freq_table
    #freq_table_json = json.dumps(freq_table)
    #print(freq_table_json)

    # Create bar chart
    plt.figure(figsize=(8, 6))  # optional, to set a custom figure size
    bars = plt.barh(range(len(category_counts)), category_counts.values, height=0.3, edgecolor='black', color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Categories')
    plt.title('Bar Chart of Categorical Data')
    plt.yticks([]) # Remove y-axis labels

     # Add data labels on top of the bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width(), i, f' {bar.get_width():.0f}', va='center')

    # Convert the plot to an image buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    plt.close()  # Close the plot to free memory
    #Keep log when bar chart is generated
    logging.info("Bar chart generated successfully.")
    return image_base64, freq_table_json

#Create function for generate descriptive statistic (returned dict)
def calculate_descriptive_statistics(data):
    df = pd.DataFrame(data, columns=['Values'])
    statistics = df.describe().to_dict()
    #Keep log when descriptive statistics is calculated
    logging.info("Descriptive statistics calculated successfully.")
    return statistics

#Create function for generate correlation (returned dict)
def pearson_correlation(data: Dict[str, List[float]]):
    keys = list(data.keys())
    df = pd.DataFrame({keys[0]: data[keys[0]], keys[1]: data[keys[1]]})
    corr_matrix = df.corr(method='pearson')
    correlation = corr_matrix.iloc[0, 1]
    result = f'Pearson correlation coefficient: {correlation:.2f}'
    #Keep log when pearson correlation is calculated
    logging.info("Pearson correlation calculated successfully.")
    return result

#Create function for generate chi-squared test (returned dict)
def chi_squared_test(data: Dict[str, List[str]]):
    df = pd.DataFrame(data)
    cols = df.columns
    contingency_table = pd.crosstab(df[cols[0]], df[cols[1]])
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    if p < 0.05:
        hypothesis = 'Reject null hypothesis' 
    else:
        hypothesis = 'Fail to reject null hypothesis'
    #Keep log when chi-squared test is calculated
    logging.info("Chi-squared test calculated successfully.")
    return {'chi2': chi2, 'p-value': p, 'degrees_of_freedom': dof, 'hypothesis': hypothesis}

#Create class for input data
class PrevalenceInput(BaseModel):
    df: Dict[str, List[str]]
    column_name: str

#Create function for generate fraility prevalence (returned dict)
def frailty_prevalence_calculation(df, column_name):
    prevalence = ((df[column_name] == str('มีสภาวะเปราะบาง')).sum() / df[column_name].count())*100
    prevalence = prevalence.round(2)
    number_of_people = int(df[column_name].count())
    return {'prevalence':prevalence, 'number_of_people':number_of_people}

#Create function for generate mmse prevalence (returned dict)
def mmse_prevalence_calculation(df, column_name):
    prevalence = ((df[column_name] == str('มีความเสี่ยงผิดปกติ')).sum() / df[column_name].count())*100
    prevalence = prevalence.round(2)
    number_of_people = int(df[column_name].count())
    return {'prevalence':prevalence, 'number_of_people':number_of_people}

#Create function for generate depression prevalence (returned dict)
def depression_prevalence_calculation(df, column_name):
    prevalence = ((df[column_name] != str('ไม่มีอาการ')).sum() / df[column_name].count())*100
    prevalence = prevalence.round(2)
    number_of_people = int(df[column_name].count())
    return {'prevalence':prevalence, 'number_of_people':number_of_people}

#Create function for generate disease co prevalence (returned dict)
def disease_co_prevalence_calculation(df, column_name):
    prevalence = ((df[column_name] != str('ไม่มีโรคร่วม')).sum() / df[column_name].count())*100
    prevalence = prevalence.round(2)
    number_of_people = int(df[column_name].count())
    return {'prevalence':prevalence, 'number_of_people':number_of_people}

#Create function for generate prevalence (returned dict)
def prevalence_caluculation(df, column_name):
    if column_name == 'สภาวะเปราะบาง':
        prevalence = frailty_prevalence_calculation(df, column_name)
        return prevalence
    elif column_name == 'เกณฑ์การประเมินแบบทดสอบสภาพสมองเบื้องต้น (MMSE-Thai)':
        prevalence = mmse_prevalence_calculation(df, column_name)
        return prevalence
    elif column_name == 'ระดับอาการโรคซึมเศร้า':
        prevalence = depression_prevalence_calculation(df, column_name)
        return prevalence
    elif column_name == 'ผลการประเมินดัชนีโรคร่วม':
        prevalence = disease_co_prevalence_calculation(df, column_name)
        return prevalence
    else:
        return "Error"
#keep log when prevalence is calculated
logging.info("Prevalence calculated successfully.")

#Create function for check data before input to generate logistic regression (returned dataframe)
def check_and_convert_categorical(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = df[col].unique()
            value_map = {value: i for i, value in enumerate(unique_values)}
            df[col] = df[col].map(value_map)
    return df

#Create function for generate logistic regression (returned dict)
def Calculate_logistic(input_data, dependent_col, independent_cols):
    df = pd.DataFrame(input_data)
    
    # Check and convert categorical data
    df = check_and_convert_categorical(df)
    print(df.head())
    # Get dependent and independent variables
    dependent_variable = df[dependent_col]
    independent_variables = df[independent_cols].copy()
    # load model
    logit_model = sm.Logit(dependent_variable, sm.add_constant(independent_variables))
    result = logit_model.fit()
    OR = np.exp(result.params)

    # Get p-values
    p_values = result.pvalues.to_dict()

    # Get standard errors
    std_err = result.bse.to_dict()

    # Get critical areas/confidence intervals
    conf_int = result.conf_int().to_dict('list')

    #keep log when logistic regression is calculated
    logging.info("Logistic regression calculated successfully.")

    # Convert result to a JSON serializable format
    return {"result": str(result), 
            "OR": OR.to_dict(), 
            "p_values": p_values,
            "std_err": std_err,
            "conf_int": conf_int
            }

#Handles validation errors for requests by logging the error and returning a 422 status code.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logging.error(f"Request validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

#crate endpoint for calculate numerical descriptive statistics
@app.post("/numerical_descriptive")
async def analyze_data(request: dict, token: str = Depends(verify_token)):
    # Extract the first (and should be the only) key in the dictionary
    data_key = next(iter(request))
    data = request[data_key]
    
    logging.info(f"Received data for analysis: {data}")
    histogram = generate_histogram(data)
    statistics = calculate_descriptive_statistics(data)
    #keep log when numerical descriptive statistics is calculated
    logging.info("Analysis completed successfully.")
    return {
        'histogram': histogram,
        'statistics': statistics
    }


# create endpoint for calculate categorical descriptive statistics
@app.post("/categorical_plot")
def analyze_categorical_data(data: Dict[str, Any], token: str = Depends(verify_token)):
    # This should log something like: Received data for analysis: {'dataaaa': ['ชาย','หญิง','ชาย','หญิง','ชาย','ชาย']}
    logging.info(f"Received data for analysis: {data}")

    # Getting the first key from the dictionary and using its value for further processing
    key = list(data.keys())[0]
    values = data.get(key)

    if not isinstance(values, list):
        raise HTTPException(status_code=400, detail="Invalid data format. Expected a list of values.")

    bar_chart, freq_table = plot_categorical_data(values)
    
    logging.info("Analysis completed successfully.")
    return {
        'bar_chart': bar_chart,
        'freq_table': freq_table
    }

#create endpoint for person correlation calculate
@app.post("/correlation")
def analyze_correlation(data: Dict[str, List[float]], token: str = Depends(verify_token)):
    logging.info(f"Received data for correlation analysis: {data}")
    correlation = pearson_correlation(data)
    #keep log when correlation is calculated
    logging.info("Correlation analysis completed successfully.")
    return {
        correlation
    }

#create endpoint for chi-squared calculate
@app.post("/chi_squared")
def perform_chi_squared_test(data: Dict[str, List[str]], token: str = Depends(verify_token)):
    logging.info(f"Received data for Chi-Squared test: {data}")
    chi_squared_result = chi_squared_test(data)
    #keep log when chi-squared test is calculated
    logging.info("Chi-Squared test completed successfully.")
    return chi_squared_result

#create endpoint for prevalence calculate
@app.post("/prevalence")
async def calculate_prevalence(input: PrevalenceInput, token: str = Depends(verify_token)):
    df = pd.DataFrame(input.df)
    column_name = input.column_name
    logging.info("Data received for prevalence calculation.")
    result = prevalence_caluculation(df, column_name)
    #keep log when prevalence is calculated
    logging.info("Prevalence calculation completed successfully.")
    return {"result": result}

#create endpoint for logistic regression calculate
@app.post("/logistic_regression")
async def perform_calculate_logistic(request: Request, token: str = Depends(verify_token)):
    data = await request.json()
    #keep log when logistic regression is received the data
    logging.info(f"Received data for Calculate logisti: {data}")

    if 'input_data' not in data or 'dependent_variable' not in data or 'independent_variables' not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input. Please provide 'input_data', 'dependent_variable' and 'independent_variables'.",
        )
    
    input_data = data['input_data']
    dependent_variable = data['dependent_variable']
    independent_variables = data['independent_variables']

    Calculate_logisti_result = Calculate_logistic(input_data, dependent_variable, independent_variables)
    #keep log when logistic regression is calculated
    logging.info("Calculate logistic regression completed successfully.")
    return Calculate_logisti_result

#run server (You can run this file to start the server by "python main.py", port: 8000)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
