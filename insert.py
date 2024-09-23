from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import pandas as pd
df = pd.read_csv("planes.csv")
df2 =pd.read_csv("transactions.csv")
output = pysqldf("SELECT * FROM df;")
print(output)
# import ast

# # Example dictionary
# data = "{'path': 'planes.csv', 'table_name': 'planes', 'column_names': ['id', 'make', 'model', 'year', 'price'], 'data_types': ['int64', 'object', 'object', 'int64', 'int64'], 'num_rows': 10}"

# # Convert string to dictionary
# data_dict = ast.literal_eval(data)

# print(data_dict)