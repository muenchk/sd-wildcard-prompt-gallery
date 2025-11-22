
from dynamicprompts.wildcards import WildcardManager
from dynamicprompts.wildcards.collection import WildcardTextFile
from dynamicprompts.wildcards.tree import WildcardTreeNode

from pathlib import Path

wildcard_manager: WildcardManager

key_wildcard:str = "wildcard"
key_prompt:str = "prompt"
key_prompt_second:str = "prompt_second"

gen_data = {}

def set_gen_data(key: str, value):
    gen_data[key] = value

def get_gen_data(key : str):
    return gen_data[key]

def erase_gen_data():
    global gen_data
    gen_data = {}


def update_wildcard_manager(path:Path):
    global wildcard_manager
    wildcard_manager = WildcardManager(path)

def get_wildcard_manager() -> WildcardManager:
    return wildcard_manager