import json
import os
from pathlib import Path

import datetime

key_version:str = "version"

key_defaultSettings = "defaultSettings"

key_wildcards = "wildcards"

key_galleries = "galleries"

current_json_version = 1

def get_base_dir():
    path = f"{os.getcwd()}"
    return Path(path) / "wildcard-gallery"

json_data = {}
json_data_temp = {}

def check_config_version():
    global json_data

    if key_version not in json_data:
        # json is too old for version, do complete upgrade to version 1
        temp = {}
        temp[key_wildcards] = {}
        temp[key_version] = current_json_version
        for card, obj in json_data.items():
            if (card == key_defaultSettings):
                temp[key_defaultSettings] = obj
            else:
                temp[key_wildcards][card] = obj

        json_data = temp
        # update wildcard keys
        update_wildcard_keys(0, current_json_version)
        # save json
        write_to_config()
    else:
        if json_data[key_version] < current_json_version:
            oldversion = json_data[key_version]
            json_data[key_version] = current_json_version
            # upgrade to current json version

            # update wildcard keys
            update_wildcard_keys(oldversion, current_json_version)
            # save json
            write_to_config()
            pass 

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
    global json_data
    global json_data_temp
    json_data_temp = {}
    json_data_temp[key_wildcards] = {}
    json_data_temp[key_galleries] = {}
    try:
        with open(get_base_dir() / "config.json", mode="r", encoding="utf-8") as file:
            json_data = json.load(file)
            print("[WG] loaded json database")
        now = datetime.datetime.now()
        try:
            os.makedirs(get_base_dir() / "backup")
        except Exception:
            pass
        try:
            with open(get_base_dir() / "backup" / ("config.json." + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_ " + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)), mode="w+", encoding="utf-8") as backup:
                json.dump(json_data, backup)
        except Exception as e:
            print(e)
    except Exception:
        json_data = {}
        json_data[key_version] = current_json_version
        json_data[key_wildcards] = {}
        json_data[key_galleries] = {}
        print("[WG] failed to load json database")
        
    check_config_version()

version = 9


key_height:str = "height"
key_width:str = "width"
key_prompt:str = "prompt"
key_all_prompts:str = "all_prompts"
key_prompt_second:str = "prompt_second"
key_negative_prompt:str = "negative_prompt"
key_all_negative_prompts:str = "all_negative_prompts"
key_sampler:str = "sampler_name"
key_scheduler:str = "schedule_type"
key_sampling_steps:str = "steps"
key_cfg:str = "cfg_scale"
key_distilled_cfg:str = "cfg_distilled"
key_batch_size:str = "batch_size"
key_batch_count:str = "batch_count"
key_seed:str = "seed"
key_all_seeds:str = "all_seeds"
key_seed_checkbox:str = "seed_checkbox"
key_seed_subseed:str = "subseed"
key_seed_all_subseeds:str = "all_subseeds"
key_seed_subseed_strength:str = "subseed_strength"
key_seed_resize_from_w:str = "seed_resize_from_w"
key_seed_resize_from_h:str = "seed_resize_from_h"
key_fontsize:str = "fontsize"
key_write_to_image:str = "write_to_image"
key_restore_faces:str = "restore_faces"
key_face_restoration_model:str = "face_restoration_model"
key_sd_model_name:str = "sd_model_name"
key_sd_model_hash:str = "sd_model_hash"
key_sd_vae_name:str = "sd_vae_name"
key_sd_vae_hash:str = "sd_vae_hash"
key_denoising_strength:str = "denoising_strength"
key_extra_generation_params:str = "extra_generation_params"
key_index_of_first_image:str = "index_of_first_image"
key_infotexts:str="infotexts"
key_styles:str="styles"
key_job_timestamp:str="job_timestamp"
key_clip_skip:str="clip_skip"
key_is_using_inpainting_conditioning:str="is_using_inpainting_conditioning"


key_image_array:str = "image_array"
key_img:str = "imagepath"
key_ia_filename:str = "ia_filename"
key_ia_caption:str = "ia_caption"
key_image_generation_info:str = "image_gen_info"
key_image_generation_html:str = "image_gen_html"

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


### wildcard functions

def create_new_wildcard(wildcard:str):
    if wildcard in json_data[key_wildcards]:
        print ("[WG] Found wildcard in database: " + wildcard)
        # update objects to contain all values that aren't contained so far
        obj = json_data[key_wildcards][wildcard]

        if key_version not in obj or obj[key_version] == None or obj[key_version] < version:
            # malformed object or lower version than current

            json_data[key_wildcards][wildcard] = get_default(obj)
            obj = json_data[key_wildcards][wildcard]
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
        json_data[key_wildcards][wildcard] = get_default(obj)
        return json_data[key_wildcards][wildcard]
    
def get_wildcard(wildcard:str):
    return json_data[key_wildcards][wildcard]

def update_wildcard(wildcard:str, key:str, value):
    if wildcard in json_data[key_wildcards]:
        obj = json_data[key_wildcards][wildcard]
        obj[key] = value
        json_data[key_wildcards][wildcard] = obj
    else:
        create_new_wildcard(wildcard)
        json_data[key_wildcards][wildcard][key] = value

def update_wildcard_temp(wildcard:str, key:str, value):
    if wildcard in json_data_temp[key_wildcards]:
        obj = json_data_temp[key_wildcards][wildcard]
        obj[key] = value
        json_data_temp[key_wildcards][wildcard] = obj
    else:
        obj = {}
        obj[key] = value
        json_data_temp[key_wildcards][wildcard] = obj

def delete_wildcard_temp(wildcard:str):
    if wildcard in json_data_temp[key_wildcards]:
        del json_data_temp[key_wildcards][wildcard]

def writeback_wildcard_changes():
    global json_data_temp
    for key in json_data_temp[key_wildcards].keys():
        tobj = json_data_temp[key_wildcards][key]

        obj = {}
        if (key in json_data[key_wildcards]):
            obj = json_data[key_wildcards][key]
        else:
            create_new_wildcard(key)
            obj = json_data[key_wildcards][key]

        for tkey in tobj.keys():
            val = tobj[tkey]

            obj[tkey] = val
        json_data[key_wildcards][key] = obj

    json_data_temp[key_wildcards] = {}

    write_to_config()

def update_wildcard_keys(oldversion, newversion):
    if (oldversion == 0 and newversion > oldversion):
        for wildcard, obj in json_data[key_wildcards].items():
            if "seed_subseed" in obj:
                obj[key_seed_subseed] = obj["seed_subseed"]
                del obj["seed_subseed"]
            if "seed_subseed_strength" in obj:
                obj[key_seed_subseed_strength] = obj["seed_subseed_strength"]
                del obj["seed_subseed_strength"]
            if "scheduler" in obj:
                obj[key_sampler] = obj["scheduler"]
                del obj["scheduler"]
            if "cfg" in obj:
                obj[key_cfg] = obj["cfg"]
                del obj["cfg"]
            if "sampling_steps" in obj:
                obj[key_sampling_steps] = obj["sampling_steps"]
                del obj["sampling_steps"]
            if key_image_generation_info in obj:
                if key_image_array in obj:
                    image_array = {}
                    count = 0
                    for cap, img in obj[key_image_array].items():
                        imageinfo = json.loads(obj[key_image_generation_info])
                        del imageinfo[key_all_negative_prompts]
                        del imageinfo[key_all_prompts]
                        del imageinfo[key_all_seeds]
                        del imageinfo[key_seed_all_subseeds]
                        # if there are enough infotexts for all images in array, pick the correct one
                        if len(imageinfo[key_infotexts]) == len(obj[key_image_array]):
                            imageinfo[key_infotexts] = imageinfo[key_infotexts][count]
                        else:
                            imageinfo[key_infotexts] = imageinfo[key_infotexts][0]
                        imageinfo[key_img] = img
                        image_array[cap] = imageinfo
                        count += 1
                    obj[key_image_array] = image_array
                del obj[key_image_generation_info]
                del obj[key_image_generation_html]
            obj[key_write_to_image] = True

### default settings

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


#prompt gallery functions

gallery_version = 1

def create_new_gallery(gallery:str):
    update_gallery = False
    obj = {}
    if isinstance(json_data[key_galleries], list):
        print("galleries is list???")
    if gallery in json_data[key_galleries]:
        print ("[WG] Found gallery in database " + gallery)
        obj = json_data[key_galleries][gallery]

        if key_version not in obj or obj[key_version] == None or obj[key_version] < gallery_version:
            # malformed obj or lower version than current
            obj = get_default(obj)
            update_gallery = True
        elif obj[key_version] == key_galleries:
            # no updates needed
            update_gallery= False
        else:
            # version is higher than current one, don't update
            pass
    else:
        update_gallery = True
        print("[WG] Creating new entry in database for gallery: " + gallery)
        obj = get_default(obj)

    if update_gallery == True:
        # update the obj


        json_data[key_galleries][gallery] = obj
        return json_data[key_galleries][gallery]
    else:
        return json_data[key_galleries][gallery]


def get_gallery(gallery:str):
    if (gallery in json_data[key_galleries]):
        return json_data[key_galleries][gallery]
    else:
        return {}

def update_gallery(gallery:str, key:str, value):
    if gallery in json_data[key_galleries]:
        json_data[key_galleries][gallery][key] = value
    else:
        create_new_gallery(gallery)

def rename_gallery(gallery_old:str, gallery_new:str):
    if gallery_old in json_data[key_galleries]:
        json_data[key_galleries][gallery_new] = json_data[key_galleries][gallery_old]
        del json_data[key_galleries][gallery_old]

def update_gallery_temp(gallery:str, key:str, value):
    if gallery in json_data_temp[key_galleries]:
        json_data_temp[key_galleries][gallery][key] = value
    else:
        json_data_temp[key_galleries][gallery] = {}
        json_data_temp[key_galleries][gallery][key] = value

def delete_gallery_temp(gallery:str):
    if gallery in json_data_temp:
        del json_data_temp[key_galleries][gallery]

def writeback_gallery_changes():
    global json_data_temp
    for key, tobj in json_data_temp[key_galleries].items():
        obj = {}
        if (key in json_data[key_galleries]):
            obj = json_data[key_galleries][key]
        else:
            obj = create_new_gallery(key)
        
        for tkey, val in tobj.items():
            obj[tkey] = val
        
        json_data[key_galleries][key] = obj
    
    json_data_temp[key_galleries] = {}

    write_to_config()

def get_galleries() -> list[str]:
    galleries = []
    if key_galleries in json_data:
        for key in json_data[key_galleries].keys():
            galleries.append(key)
        return galleries
    else:
        json_data[key_galleries] = {}
        return galleries