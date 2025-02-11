import os
import openai

# client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# model_name = "gemma2:2b"

DataPool = """## Sleep Data
Date: August 11, 2024
### Sleep Duration
- Total Sleep Time: 7 hours 23 minutes
- Time in Bed: 8 hours 5 minutes
- Sleep Efficiency: 91%
### Sleep Stages
- Light Sleep: 3 hours 42 minutes (50.2%)
- Deep Sleep: 1 hour 51 minutes (25.1%)
- REM Sleep: 1 hour 50 minutes (24.7%)
### Snoring
- Total Snoring Time: 42 minutes
- Snoring Episodes: 14
- Average Snoring Intensity: 48 dB
### Sleep Apnea
- Apnea-Hypopnea Index (AHI): 3.2 events/hour
- Total Apnea Events: 24
- Longest Apnea Duration: 18 seconds
### Heart Rate
- Average Heart Rate: 62 bpm
- Lowest Heart Rate: 54 bpm
- Highest Heart Rate: 78 bpm
### Other Metrics
- Room Temperature: 20.5°C
- Room Humidity: 45%
- Noise Level: Average 32 dB, Peak 58 dB
"""

template = """## Recommendations
### Recommendation 1
- Bullet A
- Bullet B
### Recommendation 2
- Bullet A
- Bullet B
### Recommendation 3
- Bullet A
- Bullet B
## Points of Concern
- Mention anomalies in data here
## Potential Issues
- Mention potential health concerns here
## Further Action
- Recommend the user to an expert if required
"""

class LLMConfig:
    def __init__(self, name, temperature):
        self.name, self.temperature = name, temperature

class MoAAgent:
    def __init__(self, agents, system_prompt, user_prompt, aggregator_agent, aggregator_prompt, client, model_name):
        self.agents = agents
        self.completion_tokens = 0
        self.agent_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.aggregator_agent = aggregator_agent
        self.aggregator_messages = [{"role": "system", "content": aggregator_prompt}]
        self.client = client
        self.model_name = model_name

    def run(self, silent=True):
        responses = [self._get_response(agent, silent) for agent in self.agents]
        
        if len(self.agents) > 1:
            templated_response = "Here are the responses from your colleagues:\n" + \
                "\n\n".join(f"Report {i}\n{response}" for i, response in enumerate(responses)) + \
                f"\nNow, please corroborate the given reports and reorganize it for clarity and consistency. Don't forget, You will be fired if you don't follow the template:\n{template}"
            self.aggregator_messages.append({"role": "user", "content": templated_response})
            final_response = self._get_response(self.aggregator_agent, silent)
        else:
            final_response = responses[0]
        
        return {"responses": responses, "final_response": final_response}

    def _get_response(self, agent, silent=True):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.agent_messages if agent != self.aggregator_agent else self.aggregator_messages,
            temperature=agent.temperature,
            max_tokens=2048 if agent == self.aggregator_agent else 1536,
            stream=False
        )
        self.completion_tokens += response.usage.completion_tokens
        result = response.choices[0].message.content
        if not silent:
            print(f"Response from {agent.name}:\n{result}\n")
        return result


def generate_markdown_reports(client, model_name, silent=True, num_agents = 2):
    other_tokens = 0

    agents = [
        LLMConfig(model_name, 0.5) for _ in range(num_agents)
    ]

    agent = MoAAgent(
        agents,
        system_prompt=(
            "You are a professional sleep doctor. Analyze the given data to write a sleep report. "
            "You MUST strictly adhere to the given markdown template, or else you will be fired. "
            "You will get sacked if you do not use MARKDOWN formatting consistent with the template. "
            "You will be demoted for dereliction of duty if you do not replace ALL {bracketed} information "
            "with your own content. Be SPECIFIC and aim to PERSONALIZE your report, rather than give general advice. "
            "Do not mention common knowledge or anything that is not tied to specific things in the data. "
            "Do not give template-style responses like [Mention possible diseases] or [Mention possible treatments]."
        ),
        user_prompt=(
            f"Here are the sleep stats for a patient. Help write a concise report of the patient's sleep health based on the data provided:\n{DataPool}\n"
            f"Remember your job. Please strictly adhere to this markdown template (do not add any other information or titles, etc.):\n{template}"
        ),
        aggregator_agent=LLMConfig(model_name, 0.5),
        aggregator_prompt=(
            "You are a professional sleep doctor tasked with improving reports written by your colleagues. "
            "Your goal is to corroborate the information and reorganize it for clarity and consistency. "
            "You MUST use the given markdown template, or else you will be fired. "
            "You will get sacked if you do not use MARKDOWN formatting consistent with the template. "
            "Keep only SPECIFIC and RELEVANT information. DO NOT INCLUDE ANY SLEEP DATA WITHIN YOUR REPORT!"
        ),
        client=client,
        model_name=model_name
    )

    agent_response = agent.run(silent)
    if not silent:
        print(f"# Final Response\n{agent_response['final_response']}\n\n")

    template_agent = LLMConfig(model_name, 0.7)
    template_agent_messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous editor reviewing a medical report. Your primary task is to ensure the report's formatting "
                "adheres strictly to the provided markdown template. Focus on correcting any grammatical errors, removing extra spaces "
                "and empty lines, while also maintaining consistent formatting. Ensure all sections from the template are present and properly formatted. "
                "Do not alter the content unless it's to fix formatting issues. Do not add any notes or comments such as 'Here is the updated report with corrections' "
                "or 'I have reviewed the report and made the necessary changes'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Please review and correct the formatting of this report:\n{agent_response['final_response']}\n\n"
                f"Ensure it strictly follows this markdown template:\n{template}\n\n"
                "Focus on fixing any formatting issues, grammatical errors, extra spaces, or extra lines. "
                "Maintain the original content as much as possible while ensuring perfect adherence to the template structure."
            )
        }
    ]
    checked_final_response = client.chat.completions.create(
        model=model_name,
        messages=template_agent_messages,
        temperature=template_agent.temperature,
        max_tokens=2048,
        stream=False
    )
    other_tokens += checked_final_response.usage.completion_tokens
    if not silent:
        print(f"# Checked Final Response\n{checked_final_response.choices[0].message.content}\n\n")

    if not silent:
        print(f"# Total Tokens Used: {agent.completion_tokens + other_tokens}")

    pre_template = """# Your Sleep Report
Please find below, your sleep report generated from your sleep data.

**Warning**: This report is not a replacement for medical diagnosis. It is a tool powered by AI to help you understand your sleep data, and identify potential areas of concern.
""" + DataPool
    full_report = pre_template + checked_final_response.choices[0].message.content

    return {
        "responses": agent_response['responses'],
        "final_response": agent_response['final_response'],
        "checked_final_response": checked_final_response.choices[0].message.content,
        "full_report": full_report,
        "agent_tokens": agent.completion_tokens,
        "other_tokens": other_tokens
    }

if __name__ == "__main__":
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    model_name = "gemma2:2b"
    reports = generate_markdown_reports(client, model_name)

    print(f"# Final Response\n{reports['final_response']}\n\n")
    print(f"# Checked Final Response\n{reports['checked_final_response']}\n\n")
    print(f"# Total Tokens Used: {reports['agent_tokens'] + reports['other_tokens']}")

    for i, response in enumerate(reports['responses'], 1):
        with open(f"./responses/response_{i}.md", "w") as f:
            f.write(response)

    with open("./responses/final_response.md", "w") as f:
        f.write(reports['checked_final_response'])

    with open("../Django/report/static/md/report.md", "w") as f:
        f.write(reports['full_report'])