import os
import ell
import openai
from AutoClient import *
from content import *

ell.init(store='./logdir', autocommit=True)

def evaluate_report(model: str, response: str, data: dict, template: str):
    @ell.simple(
        model=model.split("/")[-1],
        client=auto_client(model),
        temperature=0.1,
        max_tokens=200,
    )
    def evaluate_report_(response: str, data: dict, template: str):
        """
        Evaluate a single report based on provided data and template.

        Args:
            report (str): The report text to evaluate.
            data (dict): The relevant data for evaluation.
            template (str): The template to which the report should conform.

        Returns:
            Dict[str, int]: A dictionary containing ratings for each evaluation category.
        """
        return [ell.user(f"""Evaluate the following report based on the given data and template. Rate each category on a scale of 0 (worst) to 10 (best):
                    
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
        Style: [rating]""")]
    
    return evaluate_report_(response, data, template)

if __name__ == "__main__":
    print(evaluate_report("gpt-4o-2024-08-06", report, data, template))