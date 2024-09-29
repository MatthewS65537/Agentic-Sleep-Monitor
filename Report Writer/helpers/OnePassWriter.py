import openai
from helpers.AutoClient import *

def one_pass_writer_O1_(model: str, data: dict, template: str):
    """
    Generate a sleep report based on provided data and template using a zero-shot approach.

    Args:
        data (dict): The sleep data to analyze.
        template (str): The template to follow for the report.

    Returns:
        str: The generated sleep report.
    """
    client = auto_client("o1")
    messages = [
        {
            "role": "user",
            "content": f"""You are a professional sleep doctor. Analyze the given data to write a sleep report. 
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
relevant content derived from the sleep data analysis."""
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

def one_pass_writer(model: str, data: dict, template: str):   
    client = auto_client(model)
    
    messages = [
        {
            "role": "user",
            "content": f"""You are a professional sleep doctor. Analyze the given data to write a sleep report. You MUST strictly adhere to the given markdown template, or else you will be fired. You will get sacked if you do not use MARKDOWN formatting consistent with the template. You will be demoted for dereliction of duty if you do not replace ALL {{bracketed}} information with your own content. Be SPECIFIC and aim to PERSONALIZE your report, rather than give general advice. Do not mention common knowledge or anything that is not tied to specific things in the data. Do not give template-style responses like [Mention possible diseases] or [Mention possible treatments].
            
            Here are the sleep stats for a patient. Help write a concise report of the patient's sleep health based on the data provided:
            {data}

            Remember your job. Please strictly adhere to this markdown template (do not add any other information or titles, etc.):
            {template}

            Generate a sleep report based on the provided data and template. Ensure that your response is specific, 
            personalized, and directly tied to the given sleep data. Do not include any general advice or common knowledge 
            that is not supported by the data. Follow the template structure exactly, replacing all placeholder text with 
            relevant content derived from the sleep data analysis."""
        },
    ]
    
    if "o1-" in model:
        return one_pass_writer_O1_(model, data, template)
    else:
        response = client.chat.completions.create(
            model=model.split("/")[-1],
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
        )
        return response.choices[0].message.content