import os
from groq import Groq  
from Base import Logic

class Groq_Class(Logic):
    def __init__(self, config=None):
        super().__init__(config)
        # Use api_key from config file
        api_key = self.config.get("api_key", os.environ.get("GROQ_API_KEY"))
        self.client = Groq(api_key=api_key)  # Updated client initialization
        self.model = "llama3-70b-8192"  # Set model directly

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def generate_sql(self, question: str, **kwargs) -> str:
        # Use the super generate_sql
        sql = super().generate_sql(question, **kwargs)

        # Replace "\_" with "_"
        sql = sql.replace("\\_", "_")

        return sql

    def submit_prompt(self, prompt, **kwargs) -> str:
        chat_response = self.client.chat.completions.create(  # Updated method call
            messages=prompt,
            model=self.model,
        )

        return chat_response.choices[0].message.content