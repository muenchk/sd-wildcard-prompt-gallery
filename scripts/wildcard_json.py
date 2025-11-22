import json
import os
from pathlib import Path

def get_base_dir():
    path = f"{os.getcwd()}"
    return Path(path) / "wildcard-gallery"

json_data = {}
json_data_temp = {}

def write_to_config() -> bool:
    try:
        os.makedirs(get_base_dir())
    except Exception:
        pass
    try:
        with open(get_base_dir() / "config.json", mode="w+", encoding="utf-8") as file:
            global json_data
            json.dump(json_data, file)
    except Exception as e:
        print(e)
        return False
    return True

def read_from_config():
    json_data_temp = {}
    try:
        with open(get_base_dir() / "config.json", mode="r", encoding="utf-8") as file:
            global json_data
            json_data = json.load(file)
    except Exception:
        return {}

version = 9

key_height:str = "height"
key_width:str = "width"
key_prompt:str = "prompt"
key_prompt_second:str = "prompt_second"
key_negative_prompt:str = "negative_prompt"
key_sampler:str = "scheduler"
key_scheduler:str = "schedule_type"
key_sampling_steps:str = "sampling_steps"
key_cfg:str = "cfg"
key_distilled_cfg:str = "cfg_distilled"
key_batch_size:str = "batch_size"
key_batch_count:str = "batch_count"
key_seed:str = "seed"
key_seed_checkbox:str = "seed_checkbox"
key_seed_subseed:str = "seed_subseed"
key_seed_subseed_strength:str = "seed_subseed_strength"
key_seed_resize_from_w:str = "seed_resize_from_w"
key_seed_resize_from_h:str = "seed_resize_from_h"
key_fontsize:str = "fontsize"
key_write_to_image:str = "fontsize"

key_image_array:str = "image_array"
key_ia_filename:str = "ia_filename"
key_ia_caption:str = "ia_caption"
key_image_generation_info:str = "image_gen_info"
key_image_generation_html:str = "image_gen_html"

key_version:str = "version"

key_defaultSettings = "defaultSettings"

def get_def_val(key:str):
    if key == key_prompt:
        return ""
    elif key == key_prompt_second:
        return ""
    elif key == key_negative_prompt:
        return ""
    elif key == key_sampler:
        return "Euler a"
    elif key == key_scheduler:
        return "SGM Uniform"
    elif key == key_sampling_steps:
        return 20
    elif key == key_width:
        return 896
    elif key == key_height:
        return 1152
    elif key == key_batch_count:
        return 1
    elif key == key_batch_size:
        return 1
    elif key == key_distilled_cfg:
        return 3.5
    elif key == key_cfg:
        return 5
    elif key == key_write_to_image:
        return False
    elif key == key_fontsize:
        return 32
    elif key == key_seed:
        return -1
    elif key == key_seed_checkbox:
        return False
    elif key == key_seed_subseed:
        return -1
    elif key == key_seed_subseed_strength:
        return 0
    elif key == key_seed_resize_from_w:
        return 0
    elif key == key_seed_resize_from_h:
        return 0
    else:
        return None

def get_default(obj):

    obj[key_version] = version

    if key_prompt not in obj:
        obj[key_prompt] = get_default_value(key_prompt)
    if key_prompt_second not in obj:
        obj[key_prompt_second] = get_default_value(key_prompt_second)
    if key_negative_prompt not in obj:
        obj[key_negative_prompt] = get_default_value(key_negative_prompt)
    if key_sampler not in obj:
        obj[key_sampler] = get_default_value(key_sampler)
    if key_scheduler not in obj:
        obj[key_scheduler] = get_default_value(key_scheduler)
    if key_sampling_steps not in obj:
        obj[key_sampling_steps] = get_default_value(key_sampling_steps)
    if key_width not in obj:
        obj[key_width] = get_default_value(key_width)
    if key_height not in obj:
        obj[key_height] = get_default_value(key_height)
    if key_batch_count not in obj:
        obj[key_batch_count] = get_default_value(key_batch_count)
    if key_batch_size not in obj:
        obj[key_batch_size] = get_default_value(key_batch_size)
    if key_distilled_cfg not in obj:
        obj[key_distilled_cfg] = get_default_value(key_distilled_cfg)
    if key_cfg not in obj:
        obj[key_cfg] = get_default_value(key_cfg)
    if key_write_to_image not in obj:
        obj[key_write_to_image] = get_default_value(key_write_to_image)
    if key_fontsize not in obj:
        obj[key_fontsize] = get_default_value(key_fontsize)
    if key_seed not in obj:
        obj[key_seed] = get_default_value(key_seed)
    if key_seed_checkbox not in obj:
        obj[key_seed_checkbox] = get_default_value(key_seed_checkbox)
    if key_seed_subseed not in obj:
        obj[key_seed_subseed] = get_default_value(key_seed_subseed)
    if key_seed_subseed_strength not in obj:
        obj[key_seed_subseed_strength] = get_default_value(key_seed_subseed_strength)
    if key_seed_resize_from_w not in obj:
        obj[key_seed_resize_from_w] = get_default_value(key_seed_resize_from_w)
    if key_seed_resize_from_h not in obj:
        obj[key_seed_resize_from_h] = get_default_value(key_seed_resize_from_h)
    
    if key_image_array not in obj:
        #1234obj[key_image_array] = []
        obj[key_image_array] = {}

    if key_image_generation_info not in obj:
        obj[key_image_generation_info] = ""
    if key_image_generation_html not in obj:
        obj[key_image_generation_html] = ""

    return obj

def create_new_wildcard(wildcard:str):
    if wildcard in json_data:
        print ("[WG] Found wildcard in database: " + wildcard)
        # update objects to contain all values that aren't contained so far
        obj = json_data[wildcard]

        if key_version not in obj or obj[key_version] == None or obj[key_version] < version:
            # malformed object or lower version than current

            json_data[wildcard] = get_default(obj)
            obj = json_data[wildcard]
        elif obj[key_version] == version:
            # no updates are needed
            pass
        else:
            # version is higher than current one (this plugin may be deprecated)
            pass
        return obj
    else:
        # create an entirely new object from scratch
        print("[WG] Creating new entry in database for wildcard: " + wildcard)
        obj = {}
        json_data[wildcard] = get_default(obj)
        return json_data[wildcard]
    
def get_wildcard(wildcard:str):
    return json_data[wildcard]

def update_wildcard(wildcard:str, key:str, value):
    if wildcard in json_data:
        obj = json_data[wildcard]
        obj[key] = value
        json_data[wildcard] = obj
    else:
        create_new_wildcard(wildcard)
        json_data[wildcard][key] = value

def update_wildcard_temp(wildcard:str, key:str, value):
    if wildcard in json_data_temp:
        obj = json_data_temp[wildcard]
        obj[key] = value
        json_data_temp[wildcard] = obj
    else:
        obj = {}
        obj[key] = value
        json_data_temp[wildcard] = obj

def delete_wildcard_temp(wildcard:str):
    if wildcard in json_data_temp:
        del json_data_temp[wildcard]

def writeback_changes():
    global json_data_temp
    for key in json_data_temp.keys():
        tobj = json_data_temp[key]

        obj = {}
        if (key in json_data):
            obj = json_data[key]
        else:
            create_new_wildcard(key)
            obj = json_data[key]

        for tkey in tobj.keys():
            val = tobj[tkey]

            obj[tkey] = val
        json_data[key] = obj

    json_data_temp = {}

    write_to_config()

def set_default_value(key:str, value):
    if key_defaultSettings in json_data:
        defSett = json_data[key_defaultSettings]
        defSett[key] = value
        json_data[key_defaultSettings] = defSett

    else:
        obj = {}
        obj[key] = value
        json_data[key_defaultSettings] = obj

def get_default_value(key:str):
    if key_defaultSettings in json_data:
        if key in json_data[key_defaultSettings]:
            return json_data[key_defaultSettings][key]
    return get_def_val(key)