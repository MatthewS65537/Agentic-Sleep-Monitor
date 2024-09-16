import ell
from AutoClient import *

ell.init(store='./logdir', autocommit=True)

def zero_shot_writer(model: str, data: dict, template: str):    
    @ell.simple(
        model=model.split("/")[-1],
        client=auto_client(model),
        temperature=0.5,
        max_tokens=1024,
    )
    def zero_shot_writer_(data: dict, template: str):
        """
        Generate a sleep report based on provided data and template using a zero-shot approach.

        Args:
            data (dict): The sleep data to analyze.
            template (str): The template to follow for the report.

        Returns:
            str: The generated sleep report.
        """
        return [ell.user(f"""You are a professional sleep doctor. Analyze the given data to write a sleep report. 
You MUST strictly adhere to the given markdown template, or else you will be fired. 
You will get sacked if you do not use MARKDOWN formatting consistent with the template. 
You will be demoted for dereliction of duty if you do not replace ALL {{bracketed}} information 
with your own content. Be SPECIFIC and aim to PERSONALIZE your report, rather than give general advice. 
Do not mention common knowledge or anything that is not tied to specific things in the data. 
Do not give template-style responses like [Mention possible diseases] or [Mention possible treatments].

Here are the sleep stats for a patient. Help write a concise report of the patient's sleep health based on the data provided:
{data}

Remember your job. Please strictly adhere to this markdown template (do not add any other information or titles, etc.):
{template}

Generate a sleep report based on the provided data and template. Ensure that your response is specific, 
personalized, and directly tied to the given sleep data. Do not include any general advice or common knowledge 
that is not supported by the data. Follow the template structure exactly, replacing all placeholder text with 
relevant content derived from the sleep data analysis.""")]
    
    return zero_shot_writer_(data, template)

if __name__ == "__main__":
    from content import *
    model = "ollama/gemma2:2b"
    response = zero_shot_writer(model, data, template)
    print(response)