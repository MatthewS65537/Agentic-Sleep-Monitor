import autogen
import os

# Assume DataPool is a pre-defined string
DataPool = """## Sleep Data for User: John Doe
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

### Breathing Rate
- Average Breathing Rate: 14 breaths/minute
- Lowest Breathing Rate: 12 breaths/minute
- Highest Breathing Rate: 16 breaths/minute

### Other Metrics
- Body Temperature Variation: -0.3°C
- Room Temperature: 20.5°C
- Room Humidity: 45%
- Noise Level: Average 32 dB, Peak 58 dB

### Sleep Quality Score
- Overall Sleep Quality: 82/100

### Notes
- Restlessness detected at 2:15 AM, duration: 5 minutes
- Sleep talking episode at 4:32 AM, duration: 12 seconds
"""


# Configure the AI models
# groqllama70b_config = {
#     "model": "llama3-70b-8192",
#     "api_type" : "groq",
#     "api_key": os.environ["GROQ_API_KEY"]
# }

xiaoai_gpt4o_config = {
    "model": "gpt-4o",
    "base_url" : "https://api.xiaoai.plus/v1",
    "api_key": os.environ["XIAOAI_API_KEY"]
}

config_list = [xiaoai_gpt4o_config]

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Create the Doctor agent
doctor = autogen.AssistantAgent(
    name="Doctor",
    llm_config={
        "config_list": config_list,
    },
    system_message="You are a medical doctor. Analyze the given data and provide a diagnosis or medical opinion."
)

# Create the Critic agent
critic = autogen.AssistantAgent(
    name="Critic",
    llm_config={
        "config_list": config_list,
    },
    system_message="You are a medical critic. Review the doctor's analysis and provide constructive feedback or alternative viewpoints."
)

groupchat = autogen.GroupChat(
    agents=[doctor, critic, user_proxy],
    messages=[],
    max_round=11,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=xiaoai_gpt4o_config)

chat_result = user_proxy.initiate_chat(
    manager,
    message="""Here are the sleep stats for a patient. Help write a report of the patient's sleep health based on the data provided:\n""" + DataPool
)
