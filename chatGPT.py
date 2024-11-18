import openai
from utility import getGameData, in_meeting, get_chat_messages, clear_chat, translatePlayerColorID, allTasksDone, get_nearby_players, load_G, get_kill_list, get_num_alive_players
import time
import pyautogui
import networkx as nx
import re
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/task-solvers")
from task_utility import get_dimensions, get_screen_coords, wake, get_screen_ratio
# New Imports for ollama and groq
from groq import Groq
import requests
import json
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Adding new configuration class and enums

@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    # Ollama specific
    api_base: str = "http://localhost:11434"

class LLMManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._setup_client()
    
    def _setup_client(self):
        if self.config.provider == LLMProvider.OPENAI:
            openai.api_key = self.config.api_key
        elif self.config.provider == LLMProvider.GROQ:
            self.client = Groq(api_key=self.config.api_key)
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            if self.config.provider == LLMProvider.OPENAI:
                response = openai.ChatCompletion.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p
                )
                return response['choices'][0]['message']['content'].rstrip()
            
            elif self.config.provider == LLMProvider.GROQ:
                # Using Groq's OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p
                )
                return response.choices[0].message.content.rstrip()
            
            elif self.config.provider == LLMProvider.OLLAMA:
                # Using Ollama's chat endpoint
                response = requests.post(
                    f"{self.config.api_base}/api/chat",
                    json={
                        "model": self.config.model_name,
                        "messages": messages,
                        "options": {
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p
                        },
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    return response.json()['message']['content'].rstrip()
                else:
                    raise Exception(f"Ollama API error: {response.text}")
            
        except Exception as e:
            print(f"Error generating response with {self.config.provider.value}: {str(e)}")
            return ""


def load_llm_config() -> LLMConfig:
    """Load LLM configuration from llm_config.json"""
    default_config = {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "temperature": 1.0,
        "max_tokens": None,
        "top_p": 1.0,
        "api_base": "http://localhost:11434"
    }
    
    try:
        with open("llm_config.json", "r") as f:
            config = json.load(f)
            default_config.update(config)
    except FileNotFoundError:
        pass
    
    # Load API key if needed
    api_key = None
    if default_config["provider"] in ["openai", "groq"]:
        try:
            with open("APIkey.txt") as f:
                api_key = f.readline().rstrip()
        except FileNotFoundError:
            print(f"No API key found for {default_config['provider']}")
            raise SystemExit(0)
    
    return LLMConfig(
        provider=LLMProvider(default_config["provider"]),
        api_key=api_key,
        model_name=default_config["model_name"],
        temperature=default_config["temperature"],
        max_tokens=default_config["max_tokens"],
        top_p=default_config["top_p"],
        api_base=default_config["api_base"]
    )

# Initialize the LLM manager
config = load_llm_config()
llm_manager = LLMManager(config)

def ask_llm(prompts: List[Dict[str, str]]) -> str:
    """Unified function to interact with any configured LLM"""
    print(f"Sending prompt to {llm_manager.config.provider.value}")
    try:
        response = llm_manager.generate_response(prompts)
        print("Returned message")
        return response
    except Exception as e:
        print(f"Error in LLM response: {str(e)}")
        return ""

with open("sendDataDir.txt") as f:
    line = f.readline().rstrip()
    MEETING_PATH = line + "\\meetingData.txt"
    VOTE_TIME_PATH = line + "\\timerData.txt"
f.close()

def get_caller_color():
    with open(MEETING_PATH) as f:
        line = f.readline().rstrip()
        return translatePlayerColorID(int(line))
    
def get_dead_players():
    with open(MEETING_PATH) as f:
        f.readline().rstrip()
        line = f.readline().rstrip()
        return [translatePlayerColorID(int(x)) for x in line.strip('][').split(", ")[:-1]]
    
names_dict = {}
def get_names_dict():
    with open(MEETING_PATH) as f:
        f.readline().rstrip()
        f.readline().rstrip()
        big_long_input = f.readline().rstrip().strip('][').split(", ")
        for item in big_long_input:
            item = item.split("/")
            names_dict[item[0]] = int(item[1])
        return names_dict
    
def get_last_task():
    with open("last_task.txt") as f:
        line = f.readline().rstrip()
        return line
    
def get_last_room():
    with open("last_area.txt") as f:
        line = f.readline().rstrip()
        return line
    
def get_meeting_time():
    with open(VOTE_TIME_PATH) as f:
        time = int(f.readline())
    return time

# min difference is 46
cols_dict = {"RED" : (208, 68, 74), "BLUE" : (62, 91, 234), "GREEN" : (55, 156, 95), "PINK" : (241, 123, 217), 
             "ORANGE" : (241, 156, 70), "YELLOW" : (241, 246, 130), "BLACK" : (97, 111, 122), "WHITE" : (223, 240, 251),
             "PURPLE" : (134, 91, 214), "BROWN" : (138, 110, 83), "CYAN" : (95, 245, 245), "LIME" : (108, 244, 107),
             "MAROON" : (121, 75, 95), "ROSE" : (241, 213, 236), "BANANA" : (239, 244, 199), "GRAY" : (142, 162, 181),
             "TAN" : (162, 163, 156), "CORAL" : (226, 136, 144), "GREY" : (142, 162, 181), "SKIP" : ()}

def col_diff(col1 : tuple, col2 : tuple) -> int:
    return abs(col1[0] - col2[0]) + abs(col1[1] - col2[1]) + abs(col1[2] - col2[2])

def find_col_pos(dimensions, col : str):
    x = dimensions[0] + round(dimensions[2] / 7.38)
    y = dimensions[1] + round(dimensions[3] / 4.30)

    x_offset = round(dimensions[2] / 3.68)
    y_offset = round(dimensions[3] / 7.88)

    for i in range(3):
        for j in range(5):
            pixel = pyautogui.pixel(x + x_offset * i, y + y_offset * j)
            if col != "SKIP" and col_diff(cols_dict[col], pixel) < 30:
                return (x + x_offset * i, y + y_offset * j)
    return None

def skip(dimensions):
    wake()

    # skip
    time.sleep(0.3)
    pyautogui.click(dimensions[0] + round(dimensions[2] / 6.74), dimensions[1] + round(dimensions[3] / 1.15), duration=0.2)
    time.sleep(0.3)
    pyautogui.click(dimensions[0] + round(dimensions[2] / 3.87), dimensions[1] + round(dimensions[3] / 1.17), duration=0.2)

def vote(color : str = "SKIP"):
    dimensions = get_dimensions()
    x = dimensions[0] + round(dimensions[2] / 1.12)
    y = dimensions[1] + round(dimensions[3] / 19.6)
    wake()
    time.sleep(0.1)

    # close chat
    pyautogui.click(x,y, duration=0.3)
    time.sleep(0.5)

    pos = find_col_pos(dimensions, color)
    if pos is None:
        skip(dimensions)
    else:
        pyautogui.click(pos, duration=0.2)
        pyautogui.click(pos[0] + round(dimensions[2] / 8.07), pos[1], duration=0.2)

data = getGameData()

color : str = data['color']
role : str = data['status']
tasks : str = ' '.join(data['tasks'])
task_locations : str = ' '.join(data['task_locations'])
G = load_G("SHIP")
nearby_players = get_nearby_players(G)

tasks_prompt : str = "You finished all your tasks" if allTasksDone() else f"Your last completed task was {get_last_task()}"
dead_str : str = str(get_dead_players()).strip("][").replace("'", '')
kill_prompt : str = ""
kill_data = get_kill_list()
for kill in kill_data:
    if kill[0] == color:
        kill_prompt += kill[1] + ", "
if len(kill_prompt) > 0:
    kill_prompt = kill_prompt[:-2]
    kill_prompt = "You killed " + kill_prompt + " last round."

found_prompt = f'You found the body in {get_last_room()}.' if get_caller_color() == color and len(dead_str) != 0 else ''

meeting_start_time = time.time()
time.sleep(4.5)

try:
    location_prompt = f"Your tasks are in {task_locations}" if None not in task_locations else ""
except TypeError:
    location_prompt = ""

# Before the meeting, you were {"not near anyone" if len(nearby_players) == 0 else "near " + nearby_players}
prompts =   [
                {"role": "system", "content": 
                 re.sub(' +', ' ', f'''You are playing the game Among Us in a meeting with your crewmates. Your color is {color}.
                 {get_caller_color()} called the meeting. {"Nobody is" if len(dead_str) == 0 else dead_str + " are"} dead. {tasks_prompt}. The last room you were in was {get_last_room()}.
                 Before the meeting, you were {"not near anyone" if len(nearby_players) == 0 else "near " + str(nearby_players).strip("][")}. {kill_prompt} {found_prompt}
                 The prompts you see that are not from you, {color}, are messages from your crewmates. Your role is {role}. Your tasks are {tasks}. 
                 There are {get_num_alive_players()} players left alive.
                 {location_prompt}. Your crewmates' and your messages are identified by their color in the prompt. 
                 Reply to prompts with very few words and don't be formal. Try to only use 1 sentence, preferably an improper one. Never return more than 80 alphanumeric characters at a time.
                 Try to win by voting the impostor out. If your crewmates are agreeing on someone, go along with it unless you are sus of someone else. 
                 If your role is impostor, try to get other people voted off by calling them sus and suggesting the group vote them off.
                 If you are imposter, do not vote out your fellow imposters'''.replace('\n', ' '))
                },

                 {"role": "system", "content": "If someone says 'where' without much context, they are asking where the body was found"},
                 {"role": "system", "content": f"If someone says 'what' or '?' without much context, they are asking {get_caller_color()} why the meeting was called"},
                 #{"role": "system", "content": "If you decide to vote, respond by saying 'VOTE: {COLOR to vote}' or 'VOTE: skip' to skip"},
                 {"role": "system", "content": f"If people say {color} is sus or should be voted off, you need to defend youself."},
                 {"role": "system", "content": f"If you are the imposter, try gaslighting people"}, 
                 {"role": "system", "content": "Your responses MUST be of the form {YOUR COLOR}: {your message}. Do not respond in the form {OTHER PLAYER'S COLOR}: { message }."}
            ]

clear_chat()
seen_chats = []

dimensions = get_dimensions()

x = dimensions[0] + round(dimensions[2] / 1.12)
y = dimensions[1] + round(dimensions[3] / 19.6)
wake()
pyautogui.click(x,y, duration=0.3)
time.sleep(0.5)

x = dimensions[0] + round(dimensions[2] / 3)
y = dimensions[1] + round(dimensions[3] / 1.28)

pyautogui.click(x,y, duration=0.3)
time.sleep(0.1)

decided_to_vote : bool = False

while in_meeting() and not decided_to_vote:
    if time.time() - meeting_start_time > get_meeting_time() - 8:
        break
    is_new_chats = False
    chat_history = get_chat_messages()
    
    for chat in chat_history:
        if chat not in seen_chats:
            if f"{color}: " in chat:
                prompts.append({"role": "assistant", "content": chat})
            else:
                prompts.append({"role": "user", "content": chat})
                is_new_chats = True
            seen_chats.append(chat)

    try:
        if is_new_chats:
            pyautogui.click(x,y)
            time.sleep(0.1)
            response = ask_llm(prompts)
            new_response = " "
            for line in response.splitlines():
                if "VOTE: " in line:
                    print("Decided to vote")
                    if "skip" in line.lower():
                        decided_to_vote = True
                        break
                    decided_to_vote = True
                    break
                if f"{color}: " not in line:
                    print(f"skipped: {line}")
                    continue
                new_response += line

            response = new_response.replace(f'{color}: ', '')
            print("res: " + response)
            if len(response) <= 100:
                pyautogui.typewrite(f"{response.lower()}\n", interval=0.025)
            is_new_chats = False
            time.sleep(4)
    except openai.error.RateLimitError:
        print("Rate limit reached")
        break

get_names_dict()

while time.time() - meeting_start_time < get_meeting_time() - 12:
    time.sleep(1/15)

prompts.append({"role": "user", "content": "You have 10 seconds left to vote. How do you vote? Your response should be formatted as 'VOTE: {COLOR to vote}' or 'VOTE: skip' to skip"})
res = ask_llm(prompts)
col_array = ["RED", "BLUE", "GREEN", "PINK",
                "ORANGE", "YELLOW", "BLACK", "WHITE",
                "PURPLE", "BROWN", "CYAN", "LIME",
                "MAROON", "ROSE", "BANANA", "GRAY",
                "TAN", "CORAL", "GREY"]
c = "skip"
for color1 in col_array:
    if color1 in res.upper() and color1 != color:
        c = color1
if c == "skip":
    for name in names_dict:
        if name.lower() in res.lower() and translatePlayerColorID(names_dict[name]) != color:
            c = translatePlayerColorID(names_dict[name])
print("Vote: " + res)
print(c)
vote(c.upper())
time.sleep(10)
