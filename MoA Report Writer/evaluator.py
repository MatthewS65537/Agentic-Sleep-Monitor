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
        Evaluate the following response based on the given data and template. Rate each category on a scale of 0 (worst) to 10 (best):

        Data: {data}
        Template: {template}
        Response: {response}

        Please provide ratings for:
        1. Accuracy
        - How medically sound the response is (4)
        - How accurate the diagnosis is (4)
        - How specific the diagnosis is (2)
        2. Clarity
        - How clearly understandable the response is by a typical human. This includes issues with formatting and concision (3)
        - How well addressed are potential anomalies in the sleep data (3)
        - How clear are the solutions presented? As in how well can someone follow these directions to improve their sleep? (4)
        3. Relevancy
        - How relevant the response is given the data (2)
        - How useful the suggestions and diagnosis are (2)
        - Does the report contain anything other than a pure diagnosis? (2)
        - How possible is it for the patient to realistically utilize the suggestions? (4)
        4. Style
        - How well the response STRICTLY follows the template. (7)
        - How well balanced the content is in terms of legnth (3)


        Do not be afraid of rating low. It is critical that only the best reports be delivered to the patient.

        Do not include any other text in your response. Only the ratings in the format specified below. Refer to the rubric internally, but do not explicitly state it.

        Provide your final ratings STRICTLY in the format:
        Accuracy: [rating]
        Clarity: [rating]
        Relevancy: [rating]
        Style: [rating]
        """

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.25,
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