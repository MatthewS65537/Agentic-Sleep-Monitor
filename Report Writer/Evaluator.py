import os
import ell
import openai
from AutoClient import *

ell.init(store='./logdir', autocommit=True)

def evaluate_report(model : str, response: str, data: dict, template: str):
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
        return [ell.user(f"""Evaluate the following report based on the given data and template. Rate each category on a scale of 0 (worst) to 10 (best). Integers only.
                
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
        - How possible is it for the patient to realistically utilize the suggestions? (2)
        - Does the response recommend the patient to a specialist if needed? (2)
        4. Style
        - How well the response STRICTLY follows the template. (7)
        - How readable the response is in terms of formatting (2)
        - How balanced the content is (1)

        For your reference, a 6 in a category means that the result is acceptable. A 7 or higher is good. A 9 or 10 is excellent. A 4 or lower is poor.

        Do not make comments on the report. Only provide ratings. Format ratings as four integers separated by commas. Write the four integers in order of accuracy, clarity, relevancy, and style.""")]
    
    return evaluate_report_(response, data, template)

if __name__ == "__main__":
    import content
    print(evaluate_report("gpt-4o-2024-08-06", content.report_, content.data_, content.template_))