from distro import name
from pydantic_ai.models.google import GoogleModelSettings, GoogleModel
from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_graph import Graph, GraphRunContext, End, BaseNode
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import json
from dataclasses import dataclass
from typing import ClassVar
import re

load_dotenv("./.env")

class AnalystSettings(BaseSettings):
    gemini_api_key: str
    gemini_model_name: str
    temperature: float

class DatasetInfo(BaseModel):
    name: str
    num_records: int = Field(..., description="Number of rows in the dataset")
    num_features: int = Field(..., description="Number of columns in the dataset")
    columns: List[str] = Field(..., description="List of column names in the dataset")
    column_types: Dict[str, str] = Field(..., description="Mapping of column names to their data types")
    missing_values: Dict[str, int] = Field(..., description="Mapping of column names to count of missing values")
    categorical_columns: List[str] = Field(..., description="List of categorical column names")
    categories_per_column: Dict[str, List[Any]] = Field(..., description="Mapping of categorical column names to their unique categories")
    numeric_columns: List[str] = Field(..., description="List of numeric column names")
    

class FilesOrQuery(BaseModel):
    files: Optional[List[str]] = Field(default=None, description="List of file paths to datasets")
    query: Optional[str] = Field(default=None, description="Natural language query for analysis")

class AnalystState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    user_input: str = Field(..., description="User input which can be files or a query")
    files: Optional[List[str]] = Field(default=None, description="List of file paths to datasets")
    query: Optional[str] = Field(default=None, description="Natural language query for analysis")
    dataset_info: Optional[Dict[str, DatasetInfo]] = Field(default=None, description="Metadata information about the dataset")
    dataframe: Optional[Dict[str, Any]] = Field(default=None, description="Dataframe data stored as JSON-serializable dict")
    code: Optional[str] = Field(default=None, description="Generated code snippet for analysis")
    result: Optional[Any] = Field(default=None, description="Result of the data analysis")

class AnalyzeAgent(Agent):
    def __init__(self, output_type: BaseModel, settings: AnalystSettings):
        model_settings = GoogleModelSettings(
            temperature=settings.temperature,
        )

        provider = GoogleProvider(
            api_key=settings.gemini_api_key
        )
        model = GoogleModel(
            model_name=settings.gemini_model_name,
            provider=provider
        )

        super().__init__(
            model=model,
            output_type=output_type,
            model_settings=model_settings
        )

@dataclass
class InfoNode(BaseNode[AnalystState]):

    async def run(self, context: GraphRunContext[AnalystState]) -> "End[AnalystState] | QueryNode":
        if not context.state.files:
            return End(context.state)
        
        for file in context.state.files:
            df = pd.read_csv(file)
            name = file.split('/')[-1].split('.')[0]
            dataset_info = DatasetInfo(
                name=name,
                num_records=df.shape[0],
                num_features=df.shape[1],
                columns=df.columns.tolist(),
                column_types={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                missing_values={col: df[col].isnull().sum() for col in df.columns},
                categorical_columns=[col for col in df.select_dtypes(include=['object', 'category']).columns],
                categories_per_column={col: df[col].unique().tolist() for col in df.select_dtypes(include=['object', 'category']).columns},
                numeric_columns=[col for col in df.select_dtypes(include=['number']).columns]
            )
            if context.state.dataset_info is None:
                context.state.dataset_info = {}
            context.state.dataset_info[name] = dataset_info
            
            if context.state.dataframe is None:
                context.state.dataframe = {}
            context.state.dataframe[name] = df

        if context.state.query is not None:
            print("Returning to QueryNode")
            return QueryNode()
        else:
            return End(context.state)

@dataclass  
class ClassifyNode(BaseNode[AnalystState]):
    _agent: ClassVar[AnalyzeAgent] = None

    def __post_init__(self):
        agent = AnalyzeAgent(
            output_type=FilesOrQuery,
            settings=AnalystSettings()
        )

        @agent.system_prompt
        def system_prompt() -> str:
            return "You are data extraction assistant from user input. extract file paths and queries."
        

        ClassifyNode._agent = agent

    async def run(self, context: GraphRunContext[AnalystState]) -> "InfoNode | QueryNode":

        result = await self._agent.run(context.state.user_input)
        context.state.files = result.output.files
        context.state.query = result.output.query
        if result.output.files:
            return InfoNode()
        elif result.output.query:
            return QueryNode()
        else:
            raise ValueError("Input could not be classified as files or query.")

@dataclass       
class QueryNode(BaseNode[AnalystState]):
    _agent : ClassVar[AnalyzeAgent] = None

    def __post_init__(self):
        agent = AnalyzeAgent(
            output_type=str,
            settings=AnalystSettings()
        )

        @agent.system_prompt
        def system_prompt() -> str:
            return """
            You are a Python Data Analyst who writes precise, efficient, and well-structured Python code snippets to analyze pandas DataFrames based on natural language user queries.

            Your Capabilities

            You can interpret and process any analytical or statistical query, even if expressed in plain English, by automatically identifying:

            The intended operation (aggregation, filtering, sorting, correlation, hypothesis testing, etc.)

            The relevant dataset(s) from the dataframes dictionary

            The columns and metrics involved (for example, sales, category, profit)

            The appropriate analytical or statistical method to apply

            Natural Language Query Understanding

            Interpret free-form analytical questions such as “What is the most sold category?”

            Map natural language to the correct pandas or statistical operations

            Data Exploration and Summary

            Generate descriptive statistics (mean, median, mode, standard deviation, count, percentiles)

            Identify missing values, duplicates, and unique counts

            Explore relationships between categorical and numerical variables

            Statistical Analysis

            Perform t-tests, ANOVA, chi-square, and non-parametric tests

            Calculate correlation coefficients (Pearson, Spearman, Kendall)

            Conduct linear, logistic, and polynomial regression

            Compute effect sizes, p-values, and confidence intervals

            Distribution and Normality Analysis

            Visualize and analyze data distributions

            Perform normality tests (Shapiro-Wilk, Anderson-Darling, KS test)

            Detect skewness, kurtosis, and outliers

            Apply data transformations such as log, Box-Cox, and Yeo-Johnson

            Data Transformation and Feature Engineering

            Encode categorical features (one-hot, label, ordinal)

            Normalize or standardize numerical features

            Handle missing values, outliers, and data imbalance

            Create new or derived columns for enhanced analysis

            Time Series and Trend Analysis

            Resample data (daily, weekly, monthly)

            Identify trends, seasonality, and anomalies

            Conduct stationarity tests (ADF, KPSS)

            Build forecasting models (ARIMA, SARIMA, ETS)

            Available Tools

            You can use the following Python libraries for your analysis:

            import pandas as pd
            import numpy as np
            from scipy import stats
            import statsmodels.api as sm
            from statsmodels.formula.api import ols


            Instructions

            Don't provide misinformation.

            Read Dataset Details carefully.

            Interpret the user's analytical query.

            Identify the relevant dataset(s) and fields.

            Select the suitable analytical or statistical approach.

            Write optimized Python code compatible with pandas DataFrames.

            Store the final output in a variable named result.

            Output Format
            Always wrap your code snippets as follows:
            should not include any unnecessary text inside the code tags.(like ```python...```, ```javascript...```, etc.)

            <code>
            # Your python code snippet here
            </code>


            Examples

            Example - Correlation Analysis

            correlation = dataframes['sales'][['price', 'profit']].corr()
            result = correlation


            Example - T-test

            group1 = dataframes['customer'][dataframes['customer']['segment'] == 'A']['spend']
            group2 = dataframes['customer'][dataframes['customer']['segment'] == 'B']['spend']
            t_stat, p_value = stats.ttest_ind(group1, group2)
            result = {'t_statistic': t_stat, 'p_value': p_value}


            Example - ANOVA

            model = ols('spend ~ C(segment)', data=dataframes['customer']).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            result = anova_table
            """

        QueryNode._agent = agent

    async def run(self, context: GraphRunContext[AnalystState]) -> End[AnalystState]:

        dataframes = context.state.dataframe

        dataset_info = []

        for key, info in context.state.dataset_info.items():
            info_str = f"Dataset Name: {info.name}\n"
            info_str += f"Number of Records: {info.num_records}\n"
            info_str += f"Number of Features: {info.num_features}\n"
            info_str += f"Columns: {', '.join(info.columns)}\n"
            info_str += f"Column Types: {json.dumps(info.column_types)}\n"
            info_str += f"Missing Values: {json.dumps(info.missing_values)}\n"
            info_str += f"Categorical Columns: {', '.join(info.categorical_columns)}\n"
            info_str += f"Categories per Column: {json.dumps(info.categories_per_column)}\n"
            info_str += f"Numeric Columns: {', '.join(info.numeric_columns)}\n"
            dataset_info.append(info_str)

        dataset_info = "\n".join(dataset_info)

        dataset_first_5_rows = []
        for name, df in dataframes.items():
            dataset_first_5_rows.append(f"Dataset Name: {name}")
            dataset_first_5_rows.append(df.head().to_string())

        dataset_first_5_rows = "\n".join(dataset_first_5_rows)

        error = None

        max_attempts = 5
        attempts = 0

        while True:

            if error is None:

                user_prompt = f"""
                Analyze the provided datasets according to the following user query: {context.state.query}

                Dataset Details

                Dataset Information:
                {dataset_info}

                Dataset First 5 Rows:
                {dataset_first_5_rows}

                Available Variables

                dataframes: A dictionary containing multiple pandas DataFrames, where each key represents a dataset name and each value is a pandas DataFrame.

                dataframes = {{name: pd.DataFrame}}
                """
            else:
                user_prompt = f"""

                code generated in previous attempt:
                {context.state.code}

                The following error was encountered during data processing: {error}

                Please adjust the analysis accordingly for the user query: {context.state.query}

                Dataset Details

                Dataset Information:
                {dataset_info}

                Dataset First 5 Rows:
                {dataset_first_5_rows}

                Available Variables

                dataframes: A dictionary containing multiple pandas DataFrames, where each key represents a dataset name and each value is a pandas DataFrame.

                dataframes = {{name: pd.DataFrame}}
                """

            result = await self._agent.run(user_prompt=user_prompt)

            # Extract code from the result string
            code_match = re.search(r'<code>(.*?)</code>', result.output, re.DOTALL)
            if code_match:
                context.state.code = code_match.group(1).strip()
            else:
                # If no code tags found, assume the entire result is code
                context.state.code = result.output.strip()


            print("============================================")
            print("Generated Code:")
            print(context.state.code)
            print("============================================")

            # Execute the generated code
            try:
                # Create a local namespace with the dataframes
                local_vars = {'dataframes': dataframes, 'pd': pd, 'np': np, 'stats': stats, 'sm': sm, 'ols': ols}

                # Execute the generated code
                exec(context.state.code, local_vars)

                # Get the result from the executed code
                if 'result' in local_vars:
                    context.state.result = local_vars['result']
                    break
                else:
                    print("Code executed successfully but no 'result' variable found")
                    break

            except Exception as e:
                error = str(e)
                print("--------------------------------------------")
                print("Error during code execution:", error)
                print("--------------------------------------------")

                attempts += 1
                if attempts >= max_attempts:
                    print("Maximum attempts reached. Exiting.")
                    break

                continue


        return End(context.state)


if __name__ == "__main__":
    import asyncio
    initial_state = AnalystState(
        user_input= "Please analyze the datasets './customer_data.csv' and './sales_data.csv' for trends. Do some statistical analysis and give me a brief report. max 5 analysis."
    )
    graph = Graph(
        nodes=(
            ClassifyNode,
            InfoNode,
            QueryNode
        )
    )

    result = asyncio.run(graph.run(start_node=ClassifyNode(), state=initial_state))

    print(result.output.result)