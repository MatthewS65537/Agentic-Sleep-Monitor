import os
import openai
from typing import List, Dict

class EvalAgent:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def evaluate_response(self, response: str, data: str, template: str) -> Dict[str, int]:
        """
        Evaluates a single response based on provided data and template using OpenAI's API.

        Args:
            response (str): The response text to evaluate.
            data (str): The relevant data for evaluation.
            template (str): The template to which the response should conform.

        Returns:
            Dict[str, int]: A dictionary containing ratings for each evaluation category.
        """
        prompt = f"""
        Evaluate the following response based on the given data and template. Rate each category on a scale of 0 (worst) to 5 (best):

        Data: {data}
        Template: {template}
        Response: {response}

        Please provide ratings for:
        1. Accuracy: How medically sound the response is
        2. Clarity: How clearly understandable the response is
        3. Relevancy: How relevant the response is given the data
        4. Style: How well structured the response is

        Provide your ratings STRICTLY in the format:
        Accuracy: [rating]
        Clarity: [rating]
        Relevancy: [rating]
        Style: [rating]
        """

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a medical evaluation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1,
        )

        # Extract ratings from the response
        ratings = {}
        for line in completion.choices[0].message.content.split('\n'):
            if ':' in line:
                category, rating = line.split(':')
                ratings[category.strip()] = int(rating.strip())

        return ratings

    def evaluate_multiple_responses(self, responses: List[str], data: str, template: str) -> List[Dict[str, int]]:
        """
        Evaluates multiple responses based on provided data and template.

        Args:
            responses (List[str]): A list of response texts to evaluate.
            data (str): The relevant data for evaluation.
            template (str): The template to which responses should conform.

        Returns:
            List[Dict[str, int]]: A list of dictionaries containing ratings for each response.
        """
        evaluations = []
        for response in responses:
            evaluation = self.evaluate_response(response, data, template)
            evaluations.append(evaluation)
        return evaluations

# Example usage
if __name__ == "__main__":
    client = openai.OpenAI(base_url=os.getenv("HOLDAI_URL"), api_key=os.getenv("HOLDAI_API_KEY"))
    model_name = "gpt-4o-2024-08-06"
    agent = EvalAgent(client, model_name)
    
    data = "Patient presents with fever, cough, and shortness of breath."
    template = "Please provide a diagnosis and treatment plan in bullet points."
    responses = [
        "Diagnosis: Possible COVID-19\nTreatment: Rest, fluids, and isolation\nMonitor symptoms and seek emergency care if condition worsens",
        "The patient might have a respiratory infection. They should take some medicine.",
    ]
    
    results = agent.evaluate_multiple_responses(responses, data, template)
    
    print(results)
