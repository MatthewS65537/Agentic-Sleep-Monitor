import sys
sys.path.append("./Agent Tools/Heart Rate Monitor")
sys.path.append("./Agent Tools/Snoring Monitor")
sys.path.append("./Agent Tools/Movement Monitor")
sys.path.append("./skills")

import autogen

from autogen import AssistantAgent, UserProxyAgent

from config.llm_config import *

user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
)

from skills.play_music import *

music_list = ""
for music in list_available_music("./music"):
    music_list += f"./music/{music}\n"
music_player = AssistantAgent(
    "Music Player",
    system_message="You are a helpful assistant capable of playing music. Here is a list of the paths to all the music available:" + music_list + "Select any piece as you see fit.",
    llm_config={
        "config_list" : [groqllama8b_config]
        }
    )
music_player.register_for_llm(name="play_music", description="Plays music.")(play_music)
user_proxy.register_for_execution(name="play_music")(play_music)

from skills.check_vitals import *

nurse_agent = AssistantAgent(
    "Nurse",
    system_message="You are the nurse of a patient with sleep disorders. You are to monitor their various vitals and ensure they are in good health and report these results for the doctor.",
    llm_config={
        "config_list" : [groqllama70b_config]
        }
    )
# nurse_agent.register_for_llm(name="check_HRM", description="Checks the heart rate of the patient.")(check_HRM)
# user_proxy.register_for_execution(name="check_HRM")(check_HRM)
# nurse_agent.register_for_llm(name="check_AM", description="Checks the patient for signs of apnea, snoring, etc.")(check_AM)
# user_proxy.register_for_execution(name="check_AM")(check_AM)
# nurse_agent.register_for_llm(name="check_MM", description="Checks the patient's movement during sleep.")(check_MM)
# user_proxy.register_for_execution(name="check_MM")(check_MM)
nurse_agent.register_for_llm(name="check_vitals", description="Checks the vitals of the patient.")(check_vitals)
user_proxy.register_for_execution(name="check_vitals")(check_vitals)


from skills.report_life_danger import *

doctor_agent = AssistantAgent(
    "Doctor",
    system_message="You are a professional sleep doctor. You are to utilize the nurse's reports on a patient's vitals to diagnose and prescribe treatment for sleep sickness. The only available information that you have are the vitals and you cannot get any treatment results until tomorrow.",
    llm_config={
        "config_list" : [xiaoai_gpt4o_config]
        }
    )
doctor_agent.register_for_llm(name="report_danger", description="Sends an emergency notification for when the patient is in life-threatening danger.")(report_danger)
user_proxy.register_for_execution(name="report_danger")(report_danger)

groupchat = autogen.GroupChat(
    agents=[nurse_agent, doctor_agent, user_proxy],
    messages=[],
    max_round=20,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=groqllama70b_config)

chat_result = user_proxy.initiate_chat(
    manager,
    message="""How is the patient doing?"""
)