import stat
from click.utils import R
import gradio as gr
from modules.shared import opts, cmd_opts
from modules import shared, scripts
import modules
from modules.paths_internal import extensions_dir
from modules.ui import ordered_ui_categories, calc_resolution_hires, update_token_counter, update_negative_prompt_token_counter, create_output_panel, switch_values_symbol
from modules.ui_components import ToolButton, DropdownMulti, FormRow, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from typing import List, Tuple
import tempfile
import re
import hashlib
import json
import logging
from modules import shared
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules import sd_schedulers, sd_models  # noqa: F401
from modules import shared, ui_prompt_styles
from modules import script_callbacks, scripts, sd_samplers, ui_extra_networks, ui_toprow, progress, util, ui_tempdir, call_queue
from modules import txt2img
import modules.infotext_utils as parameters_copypaste
from modules.infotext_utils import image_from_url_text, PasteField
from packaging import version
from pathlib import Path
from typing import List, Tuple
from io import StringIO
from functools import lru_cache
from contextlib import ExitStack

import scripts.wildcard_json as wildcard_json
import scripts.wildcard_txt2img as wildcard_txt2img
import scripts.wildcard_data as wildcard_data
import scripts.wildcard_toprow as wildcard_toprow
import scripts.wildcard_settings as wildcard_settings
import scripts.prompt_gallery as prompt_gallery
import scripts.wildcard_misc as wildcard_misc

import os

from dynamicprompts.wildcards import WildcardManager
from dynamicprompts.wildcards.collection import WildcardTextFile
from dynamicprompts.wildcards.tree import WildcardTreeNode

from modules_forge import main_entry, forge_space
from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head

from modules.ui_common import OutputPanel, save_files, update_generation_info
folder_symbol = '\U0001f4c2'  # 📂

try:
    from scripts.wib import wib_db
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
    from wib import wib_db

try:
    import cv2
    opencv_installed = True
except ImportError:
    print("Image Browser: opencv is not installed. Video related actions cannot be performed.")
    opencv_installed = False

try:
    from modules import generation_parameters_copypaste as sendto
except ImportError:
    from modules import infotext_utils as sendto

try:
    from modules_forge import forge_version
    forge = True
except ImportError:
    forge = False

from PIL import Image, ImageOps, UnidentifiedImageError, ImageDraw

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_extension_base_path() -> Path:
    """
    Get the directory the extension is installed in.
    """
    path = f"{os.getcwd()}{os.sep}extensions{os.sep}sd-wildcard-prompt-gallery"
    assert Path(path).is_dir()  # sanity check
    return Path(path)

def get_wildcard_dir() -> Path:
    try:
        from modules.shared import opts
    except ImportError:  # likely not in an a1111 context
        opts = None
    wildcard_dir = getattr(opts, "wildcard_dir", None)
    if wildcard_dir is None:
        wildcard_dir = get_extension_base_path() / "wildcards"
    wildcard_dir = Path(wildcard_dir)
    try:
        wildcard_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception(f"Failed to create wildcard directory {wildcard_dir}")
    return wildcard_dir

wildcards : list[str]

def on_ui_settings():
    pass

send_img_path = {"value": ""}

gradio_min = "3.23.0"
gradio3_new_gallery_syntax = "3.39.0"
gradio4 = "4.0.0"
gradio4_new_gallery_syntax = "4.16.0"

temp_temp = os.path.join(scripts.basedir(), "temp")

refresh_symbol = '\U0001f504'  # 🔄


dummy_return = None

image_ext_list = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".svg"]
video_ext_list = [".mp4", ".mov", ".avi", ".wmv", ".flv", ".mkv", ".webm", ".mpeg", ".mpg", ".3gp", ".ogv", ".m4v"]
exif_cache = {}

def pure_path(path):
    if path == []:
        return path, 0
    match = re.search(r" \[(\d+)\]$", path)
    if match:
        path = path[:match.start()]
        depth = int(match.group(1))
    else:
        depth = 0
    path = os.path.realpath(path)
    return path, depth

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def traverse_all_files(curr_path, image_list, tab_base_tag_box, img_path_depth) -> List[Tuple[str, os.stat_result, str, int]]:
    global current_depth
    print(f"curr_path: {curr_path}")
    if curr_path == "":
        return image_list
    f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in image_ext_list or os.path.splitext(fname)[1] in video_ext_list:
            image_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            if (opts.image_browser_with_subdirs and tab_base_tag_box != "image_browser_tab_others") or (tab_base_tag_box == "image_browser_tab_all") or (tab_base_tag_box == "image_browser_tab_others" and img_path_depth != 0 and (current_depth < img_path_depth or img_path_depth < 0)):
                current_depth = current_depth + 1
                image_list = traverse_all_files(fname, image_list, tab_base_tag_box, img_path_depth)
                current_depth = current_depth - 1
    return image_list

def get_all_images(dir_name, img_path_depth):
    global current_depth
    print("get_all_images")
    current_depth = 0
    fileinfos = []
    
    fileinfos = traverse_all_files(dir_name, [], "image_browser_tab_others", img_path_depth)
    
    if opts.image_browser_scan_exif:
        with wib_db.transaction() as cursor:
            wib_db.fill_work_files(cursor, fileinfos)
    
    sort_values = {}
    exif_info = dict(exif_cache)
    if exif_info:
        sort_float = False

        if sort_float:
            fileinfos = [x for x in fileinfos if sort_values[x[0]] != "0"]
            fileinfos.sort(key=lambda x: float(sort_values[x[0]]))
            fileinfos = dict(fileinfos)
        else:
            fileinfos = dict(sorted(fileinfos, key=lambda x: natural_keys(sort_values[x[0]])))
        filenames = [finfo[0] for finfo in fileinfos]
    else:
        filenames = [finfo[0] for finfo in fileinfos]
    return filenames


def hash_image_path(image_path):
    image_path_hash = hashlib.md5(image_path.encode("utf-8")).hexdigest()
    cache_image_path = os.path.join(optimized_cache, image_path_hash + ".jpg")
    cache_video_path = os.path.join(optimized_cache, image_path_hash + "_video.jpg")
    return cache_image_path, cache_video_path 

def extract_video_frame(video_path, time, image_path):
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)  # time in seconds
    success, image = vidcap.read()
    if success:
        cv2.imwrite(image_path, image)
    return success

def get_thumbnail(image_video, image_list):
    global optimized_cache
    print(f"get_thumbnail with mode {image_video}")
    optimized_cache = os.path.join(tempfile.gettempdir(),"optimized")
    os.makedirs(optimized_cache,exist_ok=True)
    thumbnail_list = []
    for image_path in image_list:
        if (image_video == "image" and os.path.splitext(image_path)[1] in image_ext_list) or (image_video == "video" and os.path.splitext(image_path)[1] in video_ext_list):
            cache_image_path, cache_video_path = hash_image_path(image_path)
            if os.path.isfile(cache_image_path):
                thumbnail_list.append(cache_image_path)
            else:
                try:
                    if image_video == "image":
                        image = Image.open(image_path)
                    else:
                        extract_video_frame(image_path, 1, cache_video_path)
                        image = Image.open(cache_video_path)
                except OSError:
                    # If PIL cannot open the image, use the original path
                    thumbnail_list.append(image_path)
                    continue
                width, height = image.size
                left = (width - min(width, height)) / 2
                top = (height - min(width, height)) / 2
                right = (width + min(width, height)) / 2
                bottom = (height + min(width, height)) / 2
                thumbnail = image.crop((left, top, right, bottom)) if opts.image_browser_thumbnail_crop else ImageOps.pad(image, (max(width, height),max(width, height)), color="#000")
                thumbnail.thumbnail((opts.image_browser_thumbnail_size, opts.image_browser_thumbnail_size))

                if image_video == "video":
                    play_button_img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
                    play_button_draw = ImageDraw.Draw(play_button_img)
                    play_button_draw.polygon([(20, 20), (80, 50), (20, 80)], fill='white')
                    play_button_img = play_button_img.resize((50, 50))

                    button_for_img = Image.new('RGBA', thumbnail.size, (0, 0, 0, 0))
                    button_for_img.paste(play_button_img, (thumbnail.width - play_button_img.width, thumbnail.height - play_button_img.height), mask=play_button_img)
                    thumbnail_play = Image.alpha_composite(thumbnail.convert('RGBA'), button_for_img)
                    thumbnail.close()
                    thumbnail = thumbnail_play                    
                if thumbnail.mode != "RGB":
                    thumbnail = thumbnail.convert("RGB")
                try:
                    thumbnail.save(cache_image_path, "JPEG")
                    thumbnail_list.append(cache_image_path)
                except FileNotFoundError:
                    # Cannot save cache, use PIL object
                    thumbnail_list.append(thumbnail)
        else:
            thumbnail_list.append(image_path)
    return thumbnail_list

def get_image_page(img_path):

    img_path, _ = pure_path(img_path)
    filenames = get_all_images(img_path, 1)
    length = len(filenames)
    image_list = filenames

    image_browser_img_info = "[]"

    if opts.image_browser_use_thumbnail:
        thumbnail_list = get_thumbnail("image", image_list)
    else:
        thumbnail_list = image_list
    thumbnail_list = get_thumbnail("video", thumbnail_list)
    
    load_info = "<div style='color:#999; font-size:10px' align='center'>"
    load_info += f"{length} images in this directory {int(length+1)}"
    load_info += "</div>"
    return filenames,thumbnail_list,  None,json.dumps(image_list),image_browser_img_info


def read_path_recorder():
    path_recorder = wib_db.load_path_recorder()
    path_recorder_formatted = [value.get("path_display") for key, value in path_recorder.items()]
    path_recorder_formatted = sorted(path_recorder_formatted, key=lambda x: natural_keys(x.lower()))
    path_recorder_unformatted = list(path_recorder.keys())
    path_recorder_unformatted = sorted(path_recorder_unformatted, key=lambda x: natural_keys(x.lower()))

    return path_recorder, path_recorder_formatted, path_recorder_unformatted

def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown

def _format_node_for_json(
    wildcard_manager: WildcardManager,
    node: WildcardTreeNode,
) -> list[dict]:
    collections = [
        {
            "name": node.qualify_name(coll),
            "wrappedName": wildcard_manager.to_wildcard(node.qualify_name(coll)),
            "children": [],
        }
        for coll in sorted(node.collections)
    ]
    child_items = [
        {"name": name, "children": _format_node_for_json(wildcard_manager, child_node)}
        for name, child_node in sorted(node.child_nodes.items())
    ]
    return [*collections, *child_items]

def refresh_wildcards_callback():
    wildcard_manager = wildcard_data.get_wildcard_manager()
    wildcard_manager.clear_cache()
    root = wildcard_manager.tree.root
    tree = _format_node_for_json(wildcard_manager, root)
    cards = list(root.walk_full_names())
    collection_count = len(cards)
    cards.sort()
    print("Found " + str(collection_count) + " wildcards")
    #for name in cards:
    #    print(name)
    return cards
def refresh_wildcards():
    global wildcards
    wildcards = refresh_wildcards_callback()
    return gr.Dropdown.update(choices=wildcards)

wildcard:str = ""

def wildcard_selection_changed(selection_wildcard):
    if selection_wildcard is list:
        return "", "", "", "", 20, "Euler a", "SGM Uniform", -1, 896, 1152, 1, 5, 32, False, []
    global wildcard
    # before we actually change the selection, delete all temporary data
    # we have entered for the last loaded wildcard if there is any
    wildcard_json.delete_wildcard_temp(wildcard)
    # get the wildcard from the selection
    print(selection_wildcard)
    wildcard = selection_wildcard
    # either creates a new wildcar, or updates existing to current version
    obj = wildcard_json.create_new_wildcard(wildcard)
    # set wildcard to gen data
    wildcard_data.set_gen_data(wildcard_data.key_wildcard, wildcard)
    print(wildcard_data.get_gen_data(wildcard_data.key_wildcard))
    # selection_prompt
    
    # set initial gen data
    wildcard_data.set_gen_data(wildcard_json.key_sampler, obj[wildcard_json.key_sampler])   
    wildcard_data.set_gen_data(wildcard_json.key_scheduler, obj[wildcard_json.key_scheduler])
    wildcard_data.set_gen_data(wildcard_json.key_sampling_steps, obj[wildcard_json.key_sampling_steps])
    wildcard_data.set_gen_data(wildcard_json.key_seed, obj[wildcard_json.key_seed])
    wildcard_data.set_gen_data(wildcard_json.key_fontsize, obj[wildcard_json.key_fontsize])
    wildcard_data.set_gen_data(wildcard_data.key_prompt, obj[wildcard_json.key_prompt])
    wildcard_data.set_gen_data(wildcard_data.key_prompt_second, obj[wildcard_json.key_prompt_second])
    wildcard_data.set_gen_data(wildcard_json.key_write_to_image, obj[wildcard_json.key_write_to_image])

    images = []
    dir = wildcard_json.get_base_dir()
    #1234for img in obj[wildcard_json.key_image_array]:
        #1234path = os.path.join(dir, img[wildcard_json.key_ia_filename])
        #1234images.append((Image.open(path, "r", None), img[wildcard_json.key_ia_caption]))
    for cap, iobj in obj[wildcard_json.key_image_array].items():
        path = os.path.join(dir, iobj[wildcard_json.key_img])
        images.append((Image.open(path, "r", None), cap))

    image_generation_info, image_generation_html = wildcard_misc.get_initial_generation_info(obj[wildcard_json.key_image_array])

    # return data for ui
    return "__" + wildcard + "__", obj[wildcard_json.key_prompt], obj[wildcard_json.key_prompt_second], obj[wildcard_json.key_negative_prompt], obj[wildcard_json.key_sampling_steps], obj[wildcard_json.key_sampler], obj[wildcard_json.key_scheduler], obj[wildcard_json.key_seed], obj[wildcard_json.key_width], obj[wildcard_json.key_height], obj[wildcard_json.key_batch_size], obj[wildcard_json.key_cfg], obj[wildcard_json.key_fontsize], obj[wildcard_json.key_write_to_image], images, image_generation_info, image_generation_html, obj[wildcard_json.key_prompt] + "__" + wildcard + "__" + obj[wildcard_json.key_prompt_second]

def change_sampler_name(sampler_name):
    wildcard_data.set_gen_data(wildcard_json.key_sampler, sampler_name)   
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_sampler, sampler_name)

def change_scheduler(scheduler):
    wildcard_data.set_gen_data(wildcard_json.key_scheduler, scheduler)
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_scheduler, scheduler)

def change_steps(steps):
    wildcard_data.set_gen_data(wildcard_json.key_sampling_steps, steps)
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_sampling_steps, steps)
    
def change_seed(seed):
    wildcard_data.set_gen_data(wildcard_json.key_seed, seed)
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_seed, seed)
    
def change_font_size(font_size):
    wildcard_data.set_gen_data(wildcard_json.key_fontsize, font_size)
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_fontsize, font_size)

def change_selection_base_prompt(selection_base_prompt, selection_base_prompt_second, selection_prompt):
    wildcard_data.set_gen_data(wildcard_data.key_prompt, selection_base_prompt)
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_prompt, selection_base_prompt)
    return selection_base_prompt + selection_prompt + selection_base_prompt_second

def change_selection_base_prompt_second(selection_base_prompt, selection_base_prompt_second, selection_prompt):
    wildcard_data.set_gen_data(wildcard_data.key_prompt_second, selection_base_prompt_second)
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_prompt_second, selection_base_prompt_second)
    return selection_base_prompt + selection_prompt + selection_base_prompt_second

def change_selection_base_negative_prompt(selection_base_negative_prompt):
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_negative_prompt, selection_base_negative_prompt)

def change_width(width):
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_width, width) 

def change_height(height):
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_height, height)

def change_batch_size(batch_size):
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_batch_size, batch_size)

def change_batch_count(batch_count):
    pass

def change_cfg_scale(cfg_scale):
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_cfg, cfg_scale)

def change_enable_image_writing(enable_image_writing):
    wildcard_data.set_gen_data(wildcard_json.key_write_to_image, enable_image_writing)
    wildcard_json.update_wildcard_temp(wildcard, wildcard_json.key_write_to_image, enable_image_writing)


def on_ui_tabs():
    wildcard_json.read_from_config()

    script_dp = [script for script in modules.scripts.scripts_data if "dynamic_prompting" in script.path]

    wildcard_txt2img.load_scripts()

    scripts.scripts_current = wildcard_txt2img.scripts_custom
    wildcard_txt2img.scripts_custom.initialize_scripts(is_img2img=False)
    wildcard_txt2img.scripts_gallery.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False, head=canvas_head) as wildcard_gallery:
        with gr.Tab(label="Wildcard Gallery", id="wg_gallery_tab") as wg_gallery_tab:
            with gr.Row():
                with gr.Column(scale=3, min_width=300):
                    output_panel = wildcard_misc.create_wg_output_panel("wg", opts.outdir_txt2img_samples)


                    #output_panel = create_output_panel("wg", opts.outdir_txt2img_samples, None)
                with gr.Column(scale=1, min_width=300):
                    toprow = wildcard_toprow.Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box, id_part="wg", _wildcards=wildcards)

                    dummy_component = gr.Textbox(visible=False)
                    dummy_component_number = gr.Number(visible=False)

                    extra_tabs = gr.Tabs(elem_id="wg_extra_tabs", elem_classes=["extra-networks"])
                    extra_tabs.__enter__()

                    with gr.Tab("Generation", id="wg_generation") as wg_generation_tab, ResizeHandleRow(equal_height=False):
                        with ExitStack() as stack:
                            if shared.opts.txt2img_settings_accordion:
                                stack.enter_context(gr.Accordion("Open for Settings", open=False))
                            stack.enter_context(gr.Column(variant='compact', elem_id="wg_settings"))

                            wildcard_txt2img.scripts_custom.prepare_ui()

                            for category in ordered_ui_categories():
                                if category == "prompt":
                                    toprow.create_inline_toprow_prompts()

                                elif category == "dimensions":
                                    with FormRow():
                                        with gr.Column(elem_id="wg_column_size", scale=4):
                                            width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="wg_width")
                                            height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="wg_height")

                                        with gr.Column(elem_id="wg_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                            res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="wg_res_switch_btn", tooltip="Switch width/height")

                                        if opts.dimensions_and_batch_together:
                                            with gr.Column(elem_id="wg_column_batch"):
                                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="wg_batch_count")
                                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="wg_batch_size")

                                elif category == "cfg":
                                    with gr.Row():
                                        distilled_cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Distilled CFG Scale', value=3.5, elem_id="wg_distilled_cfg_scale")
                                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='CFG Scale', value=7.0, elem_id="wg_cfg_scale")
                                        cfg_scale.change(lambda x: gr.update(interactive=(x != 1)), inputs=[cfg_scale], outputs=[toprow.negative_prompt], queue=False, show_progress=False)
                                    with gr.Row():
                                        enable_image_writing = gr.Checkbox(value=False, label="Disable writing wildcard to image")
                                        font_size = gr.Slider(minimum=1, maximum=128, value=32, step=1, label="Font Size")

                                elif category == "checkboxes":
                                    with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                        pass

                                elif category == "accordions":
                                    with gr.Row(elem_id="wg_accordions", elem_classes="accordions"):
                                        with InputAccordion(False, label="Hires. fix", elem_id="wg_hr") as enable_hr:
                                            with enable_hr.extra():
                                                hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution")

                                            with FormRow(elem_id="wg_hires_fix_row1", variant="compact"):
                                                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="wg_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                                hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="wg_hires_steps")
                                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="wg_denoising_strength")

                                            with FormRow(elem_id="wg_hires_fix_row2", variant="compact"):
                                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="wg_hr_scale")
                                                hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="wg_hr_resize_x")
                                                hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="wg_hr_resize_y")

                                            with FormRow(elem_id="wg_hires_fix_row_cfg", variant="compact"):
                                                hr_distilled_cfg = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label="Hires Distilled CFG Scale", value=3.5, elem_id="wg_hr_distilled_cfg")
                                                hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label="Hires CFG Scale", value=7.0, elem_id="wg_hr_cfg")

                                            with FormRow(elem_id="wg_hires_fix_row3", variant="compact", visible=shared.opts.hires_fix_show_sampler) as hr_checkpoint_container:
                                                hr_checkpoint_name = gr.Dropdown(label='Hires Checkpoint', elem_id="hr_checkpoint", choices=["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint", scale=2)

                                                hr_checkpoint_refresh = ToolButton(value=refresh_symbol)

                                                def get_additional_modules():
                                                    modules_list = ['Use same choices']
                                                    if main_entry.module_list == {}:
                                                        _, modules = main_entry.refresh_models()
                                                        modules_list += list(modules)
                                                    else:
                                                        modules_list += list(main_entry.module_list.keys())
                                                    return modules_list

                                                modules_list = get_additional_modules()

                                                def refresh_model_and_modules():
                                                    modules_list = get_additional_modules()
                                                    return gr.update(choices=["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True)), gr.update(choices=modules_list)

                                                hr_additional_modules = gr.Dropdown(label='Hires VAE / Text Encoder', elem_id="hr_vae_te", choices=modules_list, value=["Use same choices"], multiselect=True, scale=3)

                                                hr_checkpoint_refresh.click(fn=refresh_model_and_modules, outputs=[hr_checkpoint_name, hr_additional_modules], show_progress=False)

                                            with FormRow(elem_id="wg_hires_fix_row3b", variant="compact", visible=shared.opts.hires_fix_show_sampler) as hr_sampler_container:
                                                hr_sampler_name = gr.Dropdown(label='Hires sampling method', elem_id="hr_sampler", choices=["Use same sampler"] + sd_samplers.visible_sampler_names(), value="Use same sampler")
                                                hr_scheduler = gr.Dropdown(label='Hires schedule type', elem_id="hr_scheduler", choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers], value="Use same scheduler")

                                            with FormRow(elem_id="wg_hires_fix_row4", variant="compact", visible=shared.opts.hires_fix_show_prompts) as hr_prompts_container:
                                                with gr.Column():
                                                    hr_prompt = gr.Textbox(label="Hires prompt", elem_id="hires_prompt", show_label=False, lines=3, placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", elem_classes=["prompt"])
                                                with gr.Column():
                                                    hr_negative_prompt = gr.Textbox(label="Hires negative prompt", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", elem_classes=["prompt"])

                                            hr_cfg.change(lambda x: gr.update(interactive=(x != 1)), inputs=[hr_cfg], outputs=[hr_negative_prompt], queue=False, show_progress=False)

                                        wildcard_txt2img.scripts_custom.setup_ui_for_section(category)

                                elif category == "batch":
                                    if not opts.dimensions_and_batch_together:
                                        with FormRow(elem_id="wg_column_batch"):
                                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="wg_batch_count")
                                            batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="wg_batch_size")

                                elif category == "override_settings":
                                    with FormRow(elem_id="wg_override_settings_row") as row:
                                        override_settings = create_override_settings_dropdown('wg', row)

                                elif category == "scripts":
                                    with FormGroup(elem_id="wg_script_container"):
                                        custom_inputs = wildcard_txt2img.scripts_custom.setup_ui()

                                if category not in {"accordions"}:
                                    wildcard_txt2img.scripts_custom.setup_ui_for_section(category)

                        hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]

                        for component in hr_resolution_preview_inputs:
                            event = component.release if isinstance(component, gr.Slider) else component.change

                            event(
                                fn=calc_resolution_hires,
                                inputs=hr_resolution_preview_inputs,
                                outputs=[hr_final_resolution],
                                show_progress=False,
                            )
                            event(
                                None,
                                _js="onCalcResolutionHires",inputs=hr_resolution_preview_inputs,
                                outputs=[],
                                show_progress=False,
                            )
                            

                    extra_networks_ui = ui_extra_networks.create_ui(wildcard_gallery, [wg_generation_tab], 'wg')
                    ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)

                    extra_tabs.__exit__()

            toprow.selection_refresh_wildcards.click(
                fn=refresh_wildcards,
                inputs=[], 
                outputs=[toprow.selection_wildcard]
            )

            steps = wildcard_txt2img.scripts_custom.script('Sampler').steps
            sampler_name = wildcard_txt2img.scripts_custom.script('Sampler').sampler_name
            scheduler = wildcard_txt2img.scripts_custom.script('Sampler').scheduler
            
            seed = wildcard_txt2img.scripts_custom.script('Seed').seed

            toprow.selection_wildcard.change(
                fn=wildcard_selection_changed,
                inputs=[toprow.selection_wildcard],
                outputs=[toprow.prompt_wildcard, toprow.prompt, toprow.prompt_second, toprow.negative_prompt, steps, sampler_name, scheduler, seed, width, height, batch_size, cfg_scale, font_size, enable_image_writing, output_panel.gallery, output_panel.generation_info, output_panel.infotext, toprow.prompt_combined]
            )

            sampler_name.change(
                fn=change_sampler_name,
                inputs=[sampler_name]
            )
            scheduler.change(
                fn=change_scheduler,
                inputs=[scheduler]
            )
            steps.change(
                fn=change_steps,
                inputs=[steps]
            )
            seed.change(
                fn=change_seed,
                inputs=[seed]
            )
            font_size.change(
                fn=change_font_size,
                inputs=[font_size]
            )
            width.change(
                fn=change_width,
                inputs=[width]
            )
            height.change(
                fn=change_height,
                inputs=[height]
            )
            batch_count.change(
                fn=change_batch_count,
                inputs=[batch_count]
            )
            batch_size.change(
                fn=change_batch_size,
                inputs=[batch_size]
            )
            cfg_scale.change(
                fn=change_cfg_scale,
                inputs=[cfg_scale]
            )
            enable_image_writing.change(
                fn=change_enable_image_writing,
                inputs=[enable_image_writing]
            )

            toprow.prompt.change(
                fn=change_selection_base_prompt, 
                inputs=[toprow.prompt, toprow.prompt_second, toprow.prompt_wildcard],
                outputs=[toprow.prompt_combined]
            )
            toprow.prompt_second.change(
                fn=change_selection_base_prompt_second,
                inputs=[toprow.prompt, toprow.prompt_second, toprow.prompt_wildcard],
                outputs=[toprow.prompt_combined]
            )
            toprow.negative_prompt.change(
                fn=change_selection_base_negative_prompt,
                inputs=[toprow.negative_prompt]
            )

            txt2img_inputs = [
                dummy_component,
                toprow.prompt_combined,
                toprow.negative_prompt,
                toprow.ui_styles.dropdown,
                batch_count,
                batch_size,
                cfg_scale,
                distilled_cfg_scale,
                height,
                width,
                enable_hr,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_checkpoint_name,
                hr_additional_modules,
                hr_sampler_name,
                hr_scheduler,
                hr_prompt,
                hr_negative_prompt,
                hr_cfg,
                hr_distilled_cfg,
                override_settings,
            ] + custom_inputs

            txt2img_outputs = [
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
            ]

            def testing(*args):
                print("testing this stuff")
                return [], "", "", ""

            txt2img_args = dict(
                #modules.txt2img
                fn=wrap_gradio_gpu_call(wildcard_txt2img.txt2img, extra_outputs=[None, '', '']),
                #fn=testing,
                _js="wg_submit",
                inputs=txt2img_inputs,
                outputs=txt2img_outputs,
                show_progress=False,
            )

            toprow.prompt.submit(**txt2img_args)
            toprow.submit.click(**txt2img_args)

            txt2img_samples_args = dict(
                #modules.txt2img
                fn=wrap_gradio_gpu_call(wildcard_txt2img.txt2img_samples, extra_outputs=[None, '', '']),
                _js="wg_submit",
                inputs=txt2img_inputs,
                outputs=txt2img_outputs,
                show_progress=False,
            )

            toprow.submit_samples.click(**txt2img_samples_args)

            txt2img_samples_save_args = dict(
                #modules.txt2img
                fn=wrap_gradio_gpu_call(wildcard_txt2img.txt2img_samples_save, extra_outputs=[None, '', '']),
                _js="wg_submit",
                inputs=txt2img_inputs,
                outputs=txt2img_outputs,
                show_progress=False,
            )
            toprow.submit_samples_save.click(**txt2img_samples_save_args)

            def select_gallery_image(index):
                index = int(index)
                if getattr(shared.opts, 'hires_button_gallery_insert', False):
                    index += 1
                return gr.update(selected_index=index)

            txt2img_upscale_inputs = txt2img_inputs[0:1] + [output_panel.gallery, dummy_component_number, output_panel.generation_info] + txt2img_inputs[1:]
            #output_panel.button_upscale.click(
            #    fn=wrap_gradio_gpu_call(modules.txt2img.txt2img_upscale, extra_outputs=[None, '', '']),
            #    _js="submit_txt2img_upscale",
            #    inputs=txt2img_upscale_inputs,
            #    outputs=txt2img_outputs,
            #    show_progress=False,
            #).then(fn=select_gallery_image, js="selected_gallery_index", inputs=[dummy_component], outputs=[output_panel.gallery])

            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressWG",
                inputs=[dummy_component],
                outputs=[
                    output_panel.gallery,
                    output_panel.generation_info,
                    output_panel.infotext,
                    output_panel.html_log,
                ],
                show_progress=False,
            )

            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
            toprow.token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
            
            scripts.scripts_current = None

        with gr.Tab(label="Prompt Gallery", id="wg_prompt_tab") as wg_prompt_tab:
            prompt_gallery.create_prompt_gallery_ui(wildcard_gallery, wg_prompt_tab, "wg_prompt")

        with gr.Tab(label="Settings", id="wg_settings_tab") as wg_settings_tab:
            wildcard_settings.create_settings_ui(wg_settings_tab, "wg_settings")


    return (wildcard_gallery , "Wildcard Gallery", "wildcard_gallery"),

wildcard_json.read_from_config()

wildcard_data.update_wildcard_manager(get_wildcard_dir())
wildcards = refresh_wildcards_callback()

script_callbacks.on_ui_tabs(on_ui_tabs)

script_callbacks.on_ui_settings(on_ui_settings)


scheduler_names = [x.label for x in sd_schedulers.schedulers]
wildcard_data.set_gen_data(wildcard_json.key_sampler, "Euler a")
wildcard_data.set_gen_data(wildcard_json.key_scheduler, scheduler_names[5])
wildcard_data.set_gen_data(wildcard_json.key_sampling_steps, 20)