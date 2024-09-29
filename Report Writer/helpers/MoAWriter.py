import time
from helpers.AutoClient import *
from helpers.OnePassWriter import *

def aggregator(model, responses, template, temperature=0.4, max_tokens=2048):    
    client = auto_client(model)
    
    responses_formatted = "Here are the responses from your colleagues:\n" + \
            "\n".join(f"Report {i}\n{response}" for i, response in enumerate(responses))
    
    messages = [
        {"role": "system", "content": "You are a professional sleep doctor tasked with improving reports written by your colleagues."},
        {"role": "user", "content": f"""Your goal is to corroborate the information and reorganize it for clarity and consistency. You MUST use the given markdown template, or else you will be fired. You will get sacked if you do not use MARKDOWN formatting consistent with the template.
        Please also ensure:
        1. Medical accuracy and specificity of the diagnosis.
        2. Usefulness and practicality of the suggestions.
        3. Readability of the report.
        Here are the reports from your colleagues:
        {responses_formatted}
        Now, please corroborate the given reports and reorganize it for clarity and consistency. Don't forget, You will be fired if you don't follow the template:
        {template}"""}
    ]
    
    response = client.chat.completions.create(
        model=model.split("/")[-1],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content

def template_checker(model, response, template, temperature=0.5, max_tokens=2048):
    client = auto_client(model)
    
    messages = [
        {"role": "system", "content": "You are a meticulous editor reviewing a medical report."},
        {"role": "user", "content": f"""Your primary task is to ensure the report's formatting adheres strictly to the provided markdown template. Focus on correcting any grammatical errors, removing extra spaces and empty lines, while also maintaining consistent formatting. Ensure all sections from the template are present and properly formatted. Do not alter the content unless it's to fix formatting issues. Do not add any notes or comments such as 'Here is the updated report with corrections' or 'I have reviewed the report and made the necessary changes'.
        Please review and correct the formatting of this report:\n{response}\n
        Ensure it strictly follows this markdown template:\n{template}\n
        Focus on fixing any formatting issues, grammatical errors, extra spaces, or extra lines. Maintain the original content as much as possible while ensuring perfect adherence to the template structure."""}
    ]
    
    response = client.chat.completions.create(
        model=model.split("/")[-1],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content

class MoAWriter:
    def __init__(self, model, data, template, num_agents=2):
        self.model = model
        self.data = data
        self.template = template
        self.num_agents = num_agents

    def run(self):
        start_time = time.time()
        responses = [one_pass_writer(self.model, self.data, self.template) for _ in range(self.num_agents)]
        aggregated_response = aggregator(self.model, responses, self.template)
        end_time = time.time()
        return {
            "time": end_time - start_time,
            "response": template_checker(self.model, aggregated_response, self.template)
        }
    
if __name__ == "__main__":
    import content
    writer = MoAWriter(model="ollama/gemma2:2b", data=content.data_, template=content.template_, num_agents=2)

    response = writer.run()
    print(f"Time taken: {response['time']}")
    print(f"{response['response']}")

    from Evaluator import *
    results = evaluate_report(model="gpt-4o-2024-08-06", response=response["response"], data=content.data_, template=content.template_)
    print(results)