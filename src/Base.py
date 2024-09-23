import os
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from urllib.parse import urlparse
from pandasql import sqldf
import pandas as pd
from exceptions import DependencyError, ImproperlyConfigured, ValidationError
from types_ import TrainingPlan, TrainingPlanItem
from utils import validate_config_path

class Logic(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)
        self.is_csv=False
        self.csv_df=pd.DataFrame()
    def log(self, message: str, title: str = "Info"):
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        if self.language is None:
            return ""

        return f"Respond in the {self.language} language."
    
    def execute_sql_on_csv(self, extracted_sql, csv_info):
        # Extracting the CSV path and SQL query from inputs
        csv_path = csv_info['path']
        table_name = csv_info['table_name']  # Use the specified table name
        
        # Use regex to replace the table name in the SQL query with 'df'
        sql_query = re.sub(r'from (\w+)', 'from df', extracted_sql, flags=re.IGNORECASE)
        sql_query=sql_query.replace(".csv", '')
        print("Modified SQL Query:", sql_query)
        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_path)
        pysqldf = lambda q: sqldf(q, globals())
        globals()['df'] = df
        print("Hellpooo")
        # Use sqldf to execute the SQL query on the DataFrame
        try:
            # Executing the SQL query using the DataFrame with the correct table name
            self.result_df = pysqldf(sql_query)
        except Exception as e:
            error_message = f"Error executing query: {e}"
            print(error_message)
            return error_message
        
        return sql_query,self.result_df 
    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        self.result_df=pd.DataFrame()
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        # question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        csv_list = self.get_related_csv(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            # question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            csv_list=csv_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)
        
        if 'csv' in llm_response:
            self.is_csv=True
            self.log(title="CSV Data Source", message="Identified CSV Data Source")
            # self.extract_sql(llm_response)
            sql_query = self.extract_sql(llm_response)
            related_csv = self.get_related_csv(sql_query, 1)
            metadata = related_csv[0]
            llm_response ,result = self.execute_sql_on_csv(sql_query, metadata)
            llm_response=llm_response.replace("\n"," ")
            return llm_response
        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        # question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        csv_list=csv_list,
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"


        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:

        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response

    # ----------------- Use Any Embeddings API ----------------- #
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    # ----------------- Use Any Database to Store and Retrieve Context ----------------- #
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass
    @abstractmethod
    def get_related_csv(self, question: str, n_results: int, **kwargs) -> list:
        """
        This method is used to get related CSV data to a question.

        Args:
            question (str): The question to get related CSV data for.
            n_results (int): The number of related CSV data to get.
        Returns:
            list: A list of related CSV data."""
        pass    
        
    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """
        This method is used to retrieve the training data from the retrieval layer.

        Returns:
            pd.DataFrame: A DataFrame containing the training data.
        """

        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was successfully removed, False otherwise.
        """

        pass

    # ----------------- Use Any Language Model API ----------------- #

    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt
    def add_csv_to_prompt(
        self,
        initial_prompt: str,
        csv_list: list[dict],
        max_tokens: int = 14000,
    ) -> str:
        if len(csv_list) > 0:
            initial_prompt += "\n===CSV Data\n\n"

            for csv_data in csv_list:
                csv_string = pd.DataFrame(csv_data).to_string(index=False)
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(csv_string)
                    < max_tokens
                ):
                    initial_prompt += f"{csv_string}\n\n"

        return initial_prompt
    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt

    #DONE
    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        # question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        csv_list: list,
        **kwargs,
    ):
        """
        Constructs a SQL prompt based on the provided context and question.

        Args:
            initial_prompt (str): The initial prompt to start with.
            question (str): The question for which the SQL query needs to be generated.
            ddl_list (list): A list of DDL (Data Definition Language) statements.
            doc_list (list): A list of documentation strings.
            csv_list (list): A list of CSV metadata strings.
            **kwargs: Additional keyword arguments for flexibility.

        Returns:
            list: A list of message logs containing the constructed prompt and the user question.
        """



        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
            "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_csv_to_prompt(
            initial_prompt, csv_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "=== Response Guidelines ===\n"
            "1. **Data Source Restriction**: Determine if the question relates to either `ddl` or `csv_list`. Use only one data source.\n"
            "   - If the question involves CSV data (`csv_list`), prepend the question with 'csv:' and use only the CSV data.\n"
            "   - If the question involves DDL data, use only the DDL data source.\n"
            "   - Do not mix data sources.\n"
            "2. **SQL Generation**: If the context is sufficient, generate a valid SQL query without any explanations.\n"
            "3. **Intermediate Query**: If the context is almost sufficient but requires knowledge of distinct values in a specific column, generate an intermediate SQL query to find distinct strings in that column. Prepend the query with a comment `-- intermediate_sql`.\n"
            "4. **Insufficient Context**: If the context is insufficient to generate the query, explain why it can't be generated.\n"
            "5. **SQL Compliance**: Ensure that the SQL output is compliant with the {self.dialect}, uses the correct data source (does not contain unwanted keywords), and is executable and free of syntax errors.\n"
        )


        message_log = [self.system_message(initial_prompt)]

        # for example in question_sql_list:
        #     if example is None:
        #         print("example is None")
        #     else:
        #         if example is not None and "question" in example and "sql" in example:
        #             message_log.append(self.user_message(example["question"]))
        #             message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log
    
    
    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ) -> list:
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        This method is used to submit a prompt to the language model and get a response.

        Args:
            prompt (str): The prompt to submit to the language model.

        Returns:
            str: The response from the language model.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))



    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        """
        Connect to a PostgreSQL database using the provided connection parameters.

        Args:
            host (str, optional): The hostname of the PostgreSQL server. Defaults to None.
            dbname (str, optional): The name of the PostgreSQL database. Defaults to None.
            user (str, optional): The username to connect to the PostgreSQL database. Defaults to None.
            password (str, optional): The password to connect to the PostgreSQL database. Defaults to None.
            port (int, optional): The port number on which the PostgreSQL server is listening. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the psycopg2.connect() method.

        Raises:
            DependencyError: If the required dependencies are not installed.
            ImproperlyConfigured: If any of the required connection parameters are not provided.

        Returns:
            psycopg2.extensions.connection: A connection object to the PostgreSQL database.
        """


        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
              
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your postgres database")

        if not user:
            user = os.getenv("PG_USER")

        if not user:
            raise ImproperlyConfigured("Please set your postgres user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs
            )
        except psycopg2.Error as e:
            raise ValidationError(e)

        def connect_to_db():
            return psycopg2.connect(host=host, dbname=dbname,
                        user=user, password=password, port=port, **kwargs)


        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = connect_to_db()  # Initial connection attempt
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                # Attempt to reconnect and retry the operation
                if conn:
                    conn.close()  # Ensure any existing connection is closed
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise ValidationError(e)

            except Exception as e:
                        conn.rollback()
                        raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres


    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """
        Executes the given SQL query and returns the result as a pandas DataFrame.

        Args:
            sql (str): The SQL query to execute.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The result of the SQL query as a pandas DataFrame.

        Raises:
            Exception: If no database connection is established.
        """

        raise Exception(
            "No database connection is established. Please set up a database connection before running SQL queries."
        )

    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        allow_llm_to_see_data: bool = False,
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None]
        ],
        None,
    ]:
        """
        This method is used to ask a question and get the corresponding SQL query and results.

        Args:
            question (Union[str, None]): The question to ask. If None, the user will be prompted to enter a question.
            print_results (bool): Whether to print the results. Default is True.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data. Default is False.

        Returns:
            Union[Tuple[Union[str, None], Union[pd.DataFrame, None], Union[plotly.graph_objs.Figure, None]], None]:
            A tuple containing the SQL query, the result as a pandas DataFrame, and a Plotly figure (if applicable).
            If print_results is True, returns None.
        """


        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            print(e)
            return None, None, None

        if print_results:
            try:
                
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
                print("CODE")
            except Exception as e:
                print(sql)

        if self.run_sql_is_set is False:
            print(
                "If you want to run the SQL query, connect to a database first."
            )

            if print_results:
                return None
            else:
                return sql, None, None

        try:
            if self.is_csv:
                df = self.result_df
            else:
                print("CODE 2",sql)
                df = self.run_sql(sql)

            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromList=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print(df)

            # Only generate plotly code if visualize is True

            return sql, df

        except Exception as e:

            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, None
        return sql, df

    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        csv: str = None,
        documentation: str = None,
        plan: TrainingPlan = None
    ) -> str:
        """
        This method is used to train the model with various types of data including questions with SQL queries, DDL statements, CSV files, and documentation.

        Args:
            question (str, optional): The question to be associated with the SQL query.
            sql (str, optional): The SQL query to be added.
            ddl (str, optional): The DDL statement to be added.
            csv (str, optional): The path to the CSV file to be added.
            documentation (str, optional): The documentation to be added.
            plan (TrainingPlan, optional): The training plan containing multiple items to be added.

        Returns:
            str: The ID of the added item.

        Raises:
            ValidationError: If a question is provided without a corresponding SQL query.
        """


        if question and not sql:
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                print("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl)
        if csv:
            print("Adding csv:", csv)
            return self.add_csv(csv)
        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_CSV:
                    self.add_csv(item.item_value)

    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                print(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES")

        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        """
        This method is used to generate a training plan from an information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS into groups of table/column descriptions that can be used to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column,
                    schema_column,
                    table_column]
        candidates = ["column_name",
                      "data_type",
                      "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[columns].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan
