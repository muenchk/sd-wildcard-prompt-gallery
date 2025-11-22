from socket import NI_NAMEREQD
import stat
from click.utils import R
import gradio as gr
from modules.shared import opts, cmd_opts
from modules import shared, scripts
from modules.paths_internal import extensions_dir
from modules.ui_components import ToolButton, DropdownMulti
from modules import script_callbacks
from typing import List, Tuple
import tempfile
import re
import hashlib
import json
import logging
from modules import shared
from modules.call_queue import wrap_gradio_gpu_call
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from modules import sd_schedulers, sd_models  # noqa: F401
from modules import shared, ui_prompt_styles
from modules import script_callbacks, scripts, sd_samplers
from modules import txt2img
from packaging import version
from pathlib import Path
from typing import List, Tuple
from io import StringIO
from functools import lru_cache

import scripts.wildcard_json as wildcard_json
import scripts.wildcard_txt2img as wildcard_txt2img
import scripts.wildcard_data as wildcard_data
import scripts.wildcard_toprow as wildcard_toprow

from modules.ui import ordered_ui_categories, calc_resolution_hires, update_token_counter, update_negative_prompt_token_counter, create_output_panel, switch_values_symbol
from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
from modules import ui_extra_networks, ui_toprow, progress, util, ui_tempdir, call_queue
import modules.infotext_utils as parameters_copypaste
from modules.infotext_utils import image_from_url_text, PasteField
from contextlib import ExitStack
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
import modules.txt2img
from modules_forge import main_entry, forge_space

from modules.ui_common import OutputPanel, save_files
import modules.ui as ui

def save_default_settings(default_prompt, default_prompt_second, default_neg_prompt, default_sampler, default_scheduler, default_steps, default_width, default_height, default_batch_count, default_batch_size, default_distilled_cfg, default_cfg, default_image_writing, default_font_size, default_seed, default_seed_checkbox, default_subseed, default_subseed_strength, default_seed_resize_from_w, default_seed_resize_from_h):

    wildcard_json.set_default_value(wildcard_json.key_prompt, default_prompt)
    wildcard_json.set_default_value(wildcard_json.key_prompt_second, default_prompt_second)
    wildcard_json.set_default_value(wildcard_json.key_negative_prompt, default_neg_prompt)
    wildcard_json.set_default_value(wildcard_json.key_sampler, default_sampler)
    wildcard_json.set_default_value(wildcard_json.key_scheduler, default_scheduler)
    wildcard_json.set_default_value(wildcard_json.key_sampling_steps, default_steps)
    wildcard_json.set_default_value(wildcard_json.key_width, default_width)
    wildcard_json.set_default_value(wildcard_json.key_height, default_height)
    wildcard_json.set_default_value(wildcard_json.key_batch_count, default_batch_count)
    wildcard_json.set_default_value(wildcard_json.key_batch_size, default_batch_size)
    wildcard_json.set_default_value(wildcard_json.key_distilled_cfg, default_distilled_cfg)
    wildcard_json.set_default_value(wildcard_json.key_cfg, default_cfg)
    wildcard_json.set_default_value(wildcard_json.key_write_to_image, default_image_writing)
    wildcard_json.set_default_value(wildcard_json.key_fontsize, default_font_size)
    wildcard_json.set_default_value(wildcard_json.key_seed, default_seed)
    wildcard_json.set_default_value(wildcard_json.key_seed_checkbox, default_seed_checkbox)
    wildcard_json.set_default_value(wildcard_json.key_seed_subseed, default_subseed)
    wildcard_json.set_default_value(wildcard_json.key_seed_subseed_strength, default_subseed_strength)
    wildcard_json.set_default_value(wildcard_json.key_seed_resize_from_w, default_seed_resize_from_w)
    wildcard_json.set_default_value(wildcard_json.key_seed_resize_from_h, default_seed_resize_from_h)

    if wildcard_json.write_to_config() == False:
        return "Config file cannot be written"
    else:
        return ""
    
def load_default_values():
    return wildcard_json.get_default_value(wildcard_json.key_prompt), \
    wildcard_json.get_default_value(wildcard_json.key_prompt_second),\
    wildcard_json.get_default_value(wildcard_json.key_negative_prompt),\
    wildcard_json.get_default_value(wildcard_json.key_sampler),\
    wildcard_json.get_default_value(wildcard_json.key_scheduler),\
    wildcard_json.get_default_value(wildcard_json.key_sampling_steps),\
    wildcard_json.get_default_value(wildcard_json.key_width),\
    wildcard_json.get_default_value(wildcard_json.key_height),\
    wildcard_json.get_default_value(wildcard_json.key_batch_count),\
    wildcard_json.get_default_value(wildcard_json.key_batch_size),\
    wildcard_json.get_default_value(wildcard_json.key_distilled_cfg),\
    wildcard_json.get_default_value(wildcard_json.key_cfg),\
    wildcard_json.get_default_value(wildcard_json.key_write_to_image),\
    wildcard_json.get_default_value(wildcard_json.key_fontsize),\
    wildcard_json.get_default_value(wildcard_json.key_seed),\
    wildcard_json.get_default_value(wildcard_json.key_seed_checkbox),\
    wildcard_json.get_default_value(wildcard_json.key_seed_subseed),\
    wildcard_json.get_default_value(wildcard_json.key_seed_subseed_strength),\
    wildcard_json.get_default_value(wildcard_json.key_seed_resize_from_w),\
    wildcard_json.get_default_value(wildcard_json.key_seed_resize_from_h)

def create_settings_ui(tab, id_part):
    with gr.Tab(label="Default Generation", id="wg_settings_defgen_tab") as wg_default_settings_tab:
        tabname = "wg_settings_defgen_tab"
        default_prompt = None
        default_prompt_second = None
        default_neg_prompt = None
        default_sampler = None
        default_scheduler = None
        default_steps = None
        default_width = None
        default_height = None
        default_batch_count = None
        default_batch_size = None
        default_distilled_cfg = None
        default_cfg = None
        default_image_writing = None
        default_font_size = None
        default_seed = None

        default_seed_checkbox = None
        default_subseed = None
        default_subseed_strength = None
        default_seed_resize_from_w = None
        default_seed_resize_from_h = None

        save_settings = None

        # warning box
        with gr.Row():
            warning_box = gr.Label(value="", show_label=False)
            
        # prompts
        with gr.Row():
            default_prompt = gr.Textbox(label="Beginning of prompt", elem_id=f"{id_part}_prompt", show_label=True, lines=3, placeholder="Prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)", elem_classes=["prompt"], value=wildcard_json.get_default_value(wildcard_json.key_prompt))
        with gr.Row():
            default_prompt_second = gr.Textbox(label="End of prompt", elem_id=f"{id_part}_prompt_second", show_label=True, lines=3, placeholder="Prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)", elem_classes=["prompt"], value=wildcard_json.get_default_value(wildcard_json.key_prompt_second))
        with gr.Row():
            default_neg_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=True, lines=3, placeholder="Negative prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)", elem_classes=["prompt"], value=wildcard_json.get_default_value(wildcard_json.key_negative_prompt))
        # sampling
        with gr.Row():
            sampler_names = [x.name for x in sd_samplers.visible_samplers()]
            scheduler_names = [x.label for x in sd_schedulers.schedulers]
            if shared.opts.samplers_in_dropdown:
                with FormRow(elem_id=f"sampler_selection_{tabname}"):
                    default_sampler = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=sampler_names, value=wildcard_json.get_default_value(wildcard_json.key_sampler))
                    default_scheduler = gr.Dropdown(label='Schedule type', elem_id=f"{tabname}_scheduler", choices=scheduler_names, value=wildcard_json.get_default_value(wildcard_json.key_scheduler))
                    default_steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=wildcard_json.get_default_value(wildcard_json.key_sampling_steps))
            else:
                with FormGroup(elem_id=f"sampler_selection_{tabname}"):
                    default_steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=wildcard_json.get_default_value(wildcard_json.key_sampling_steps))
                    default_sampler = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=sampler_names, value=wildcard_json.get_default_value(wildcard_json.key_sampler))
                    default_scheduler = gr.Dropdown(label='Schedule type', elem_id=f"{tabname}_scheduler", choices=scheduler_names, value=wildcard_json.get_default_value(wildcard_json.key_scheduler))
        # dimensions, count
        with FormRow():
            with gr.Column(elem_id=f"{tabname}_size", scale=4):
                default_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=wildcard_json.get_default_value(wildcard_json.key_width), elem_id=f"{tabname}_width")
                default_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=wildcard_json.get_default_value(wildcard_json.key_height), elem_id=f"{tabname}_height")

            with gr.Column(elem_id=f"{tabname}_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id=f"{tabname}res_switch_btn", tooltip="Switch width/height")

            if opts.dimensions_and_batch_together:
                with gr.Column(elem_id=f"{tabname}_column_batch"):
                    default_batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=wildcard_json.get_default_value(wildcard_json.key_batch_count), elem_id=f"{tabname}_batch_count")
                    default_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=wildcard_json.get_default_value(wildcard_json.key_batch_size), elem_id=f"{tabname}_batch_size")
        # cfg
        with gr.Row():
            default_distilled_cfg = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Distilled CFG Scale', value=wildcard_json.get_default_value(wildcard_json.key_distilled_cfg), elem_id="wg_distilled_cfg_scale")
            default_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='CFG Scale', value=wildcard_json.get_default_value(wildcard_json.key_cfg), elem_id="wg_cfg_scale")
            default_cfg.change(lambda x: gr.update(interactive=(x != 1)), inputs=[default_cfg], outputs=[default_neg_prompt], queue=False, show_progress=False)
        # image writing
        with gr.Row():
            default_image_writing = gr.Checkbox(value=wildcard_json.get_default_value(wildcard_json.key_write_to_image), label="Disable writing wildcard to image")
            default_font_size = gr.Slider(minimum=1, maximum=128, value=wildcard_json.get_default_value(wildcard_json.key_fontsize), step=1, label="Font Size")
        #seed
        with gr.Group():
            with gr.Row(elem_id="seed_row"):
                if cmd_opts.use_textbox_seed:
                    default_seed = gr.Textbox(label='Seed', value=wildcard_json.get_default_value(wildcard_json.key_seed), elem_id="seed", min_width=100)
                else:
                    default_seed = gr.Number(label='Seed', value=wildcard_json.get_default_value(wildcard_json.key_seed), elem_id="seed", min_width=100, precision=0)

                random_seed = ToolButton(ui.random_symbol, elem_id="random_seed", tooltip="Set seed to -1, which will cause a new random number to be used every time")

                default_seed_checkbox = gr.Checkbox(label='Extra', elem_id="subseed_show", value=wildcard_json.get_default_value(wildcard_json.key_seed_checkbox), scale=0, min_width=60)

            with gr.Group(visible=False, elem_id="seed_extras") as seed_extras:
                with gr.Row(elem_id="subseed_row"):
                    default_subseed = gr.Number(label='Variation seed', value=wildcard_json.get_default_value(wildcard_json.key_seed_subseed), elem_id="subseed", precision=0)
                    random_subseed = ToolButton(ui.random_symbol, elem_id="random_subseed")
                    reuse_subseed = ToolButton(ui.reuse_symbol, elem_id="reuse_subseed")
                    default_subseed_strength = gr.Slider(label='Variation strength', value=wildcard_json.get_default_value(wildcard_json.key_seed_subseed_strength), minimum=0, maximum=1, step=0.01, elem_id="subseed_strength")

                with gr.Row(elem_id="seed_resize_from_row"):
                    default_seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=wildcard_json.get_default_value(wildcard_json.key_seed_resize_from_w), elem_id="seed_resize_from_w")
                    default_seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=wildcard_json.get_default_value(wildcard_json.key_seed_resize_from_h), elem_id="seed_resize_from_h")

            random_seed.click(fn=None, _js="function(){setRandomSeed('" + "seed" + "')}", show_progress=False, inputs=[], outputs=[])
            random_subseed.click(fn=None, _js="function(){setRandomSeed('" + "subseed" + "')}", show_progress=False, inputs=[], outputs=[])

            default_seed_checkbox.change(lambda x: gr.update(visible=x), show_progress=False, inputs=[default_seed_checkbox], outputs=[seed_extras])

        # save button
        with gr.Row():
            save_settings = gr.Button("Save Default Settings", elem_id=f"{tabname}_save", variant='primary')

    tab.select(
        fn=load_default_values,
        inputs=[],
        outputs=[default_prompt, default_prompt_second, default_neg_prompt, default_sampler, default_scheduler, default_steps, default_width, default_height, default_batch_count, default_batch_size, default_distilled_cfg, default_cfg, default_image_writing, default_font_size, default_seed, default_seed_checkbox, default_subseed, default_subseed_strength, default_seed_resize_from_w, default_seed_resize_from_h]
    )

    save_settings.click(
        fn=save_default_settings,
        inputs=[default_prompt, default_prompt_second, default_neg_prompt, default_sampler, default_scheduler, default_steps, default_width, default_height, default_batch_count, default_batch_size, default_distilled_cfg, default_cfg, default_image_writing, default_font_size, default_seed, default_seed_checkbox, default_subseed, default_subseed_strength, default_seed_resize_from_w, default_seed_resize_from_h],
        outputs=[warning_box]
    )
        
        
        