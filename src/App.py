from flask import Flask, request, jsonify, render_template
from ChromaDB import ChromaDB_VectorStore
from LLM import Groq_Class

app = Flask(__name__)

class App(ChromaDB_VectorStore, Groq_Class):
    def __init__(self):
        ChromaDB_VectorStore.__init__(self)
        Groq_Class.__init__(self)

app_instance = App()
app_instance.connect_to_postgres(host='localhost', dbname='indexer', user='postgres', password='Sarthak@123', port='5432')

@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    csv_input = request.args.get('csv')  # Get the CSV parameter from the request
    if csv_input:
        app_instance.train(csv=csv_input)  # Use the train method with CSV input
    else:
        df_information_schema = app_instance.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        plan = app_instance.get_training_plan_generic(df_information_schema)
        app_instance.train(plan=plan)  # Use the existing training method
    return jsonify({"message": "Training completed successfully"})

@app.route('/ask', methods=['POST'])  # Change POST to GET
def ask():
    data = request.get_json()  # Use JSON body
    question = data.get('question')
    query, result_df = app_instance.ask(question=question)
    return jsonify({"query": query, "result_df": result_df.to_dict()})

if __name__ == '__main__':
    app.run(debug=True)