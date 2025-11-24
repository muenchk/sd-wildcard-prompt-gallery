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
import scripts.wildcard_misc as wildcard_misc

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
from PIL import Image

import os

def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown

def get_all_galleries() -> List[str]:
    return wildcard_json.get_galleries()

def refresh_galleries_():
    global galleries
    galleries = get_all_galleries()


folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄



galleries:list[str] = []
gallery_selected:str = ""

def get_gallery_name()-> str:
    return gallery_selected

def change_gallery_selection(gallery_selection):
    global gallery_selected
    gallery_selected = gallery_selection
    # find gallery object
    obj = wildcard_json.create_new_gallery(gallery_selected)
    # get images
    images = []
    dir = wildcard_json.get_base_dir() / "galleries"
    for cap, img in obj[wildcard_json.key_image_array].items():
        path = os.path.join(dir, img)
        images.append((Image.open(path, "r", None), cap))

    # build geninfo json data
    image_generation_info, image_generation_html = wildcard_misc.get_initial_generation_info(obj[wildcard_json.key_image_array])

    # return data 
    return gallery_selection, obj[wildcard_json.key_prompt], obj[wildcard_json.key_negative_prompt], obj[wildcard_json.key_sampling_steps], obj[wildcard_json.key_sampler], obj[wildcard_json.key_scheduler], obj[wildcard_json.key_seed], obj[wildcard_json.key_width], obj[wildcard_json.key_height], obj[wildcard_json.key_batch_count], obj[wildcard_json.key_batch_size], obj[wildcard_json.key_cfg], obj[wildcard_json.key_distilled_cfg], images, obj[wildcard_json.key_image_generation_info], obj[wildcard_json.key_image_generation_html]

def click_refresh_galleries():
    refresh_galleries_()
    return galleries

def click_add_gallery(gallery_name):
    # add gallery to gallery list and update list
    wildcard_json.create_new_gallery(gallery_name)
    refresh_galleries_()
    global gallery_selected
    gallery_selected = gallery_name
    return gr.Dropdown.update(choices=galleries), gallery_name

def click_rename_gallery(gallery_name):
    global gallery_selected
    # just rename object in json, and get new galleries list
    wildcard_json.rename_gallery(gallery_selected, gallery_name)
    refresh_galleries_()
    gallery_selected = gallery_name
    return gr.Dropdown.update(choices=galleries), gallery_name

def change_width(width):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_width, width)

def change_height(height):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_height, height)

def change_batch_count(batch_count):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_batch_count, batch_count)

def change_batch_size(batch_size):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_batch_size, batch_size)
def change_distilled_cfg_scale(distilled_cfg_scale):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_distilled_cfg, distilled_cfg_scale)
def change_cfg_scale(cfg_scale):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_cfg, cfg_scale)
    return gr.update(interactive=(cfg_scale != 1))
def change_prompt(prompt):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_prompt, prompt)
def change_negative_prompt(negative_prompt):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_negative_prompt, negative_prompt)
def change_steps(steps):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_sampling_steps, steps)
def change_sampler_name(sampler_name):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_sampler, sampler_name)
def change_scheduler(scheduler):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_scheduler, scheduler)
def change_seed(seed):
    wildcard_json.update_gallery_temp(gallery_selected, wildcard_json.key_seed, seed)

def create_prompt_gallery_ui(blocks, tab, id_part):

    refresh_galleries_()

    scripts.scripts_current = wildcard_txt2img.scripts_gallery
    wildcard_txt2img.scripts_gallery.initialize_scripts(is_img2img=False)

    with gr.Row():
        # gallery selection
        gallery_selection = gr.Dropdown(value=galleries[0] if len(galleries) > 0 else "Empty", choices=galleries, label="Select Gallery", elem_id=f"{id_part}_gallery_dropdown")
        refresh_galleries = ToolButton(value=refresh_symbol, elem_id=f"{id_part}_gallery_refresh")
        gallery_name = gr.Textbox(elem_id=f"{id_part}_gallery_name", label="Gallery Name", show_label=True, lines=1, value='')
        add_gallery = gr.Button(value="Add new gallery", elem_id=f"{id_part}_add_gallery")
        rename_gallery = gr.Button(value="Rename gallery", elem_id=f"{id_part}_rename_gallery")
    with gr.Row():
        with gr.Column(scale = 3, min_width=300):
            output_panel = wildcard_misc.create_wg_output_panel("wg_prompt", opts.outdir_txt2img_samples)

        with gr.Column(scale=1, min_width=300):
            with gr.Tab("Gallery Options", id=f"{id_part}_gallery_options"):
                # gallery options, adding removing images, etc.
                pass
            with gr.Tab("Gallery Generation", id=f"{id_part}_gallery_generation"):
                # for generating new gallery elements directly from the tab
                toprow = ui_toprow.Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box, id_part=id_part)

                dummy_component = gr.Textbox(visible=False)
                dummy_component_number = gr.Number(visible=False)

                extra_tabs = gr.Tabs(elem_id=f"{id_part}_extra_tabs", elem_classes=["extra_networks"])
                extra_tabs.__enter__()

                with gr.Tab("Generation", id=f"{id_part}_generation") as prompt_generation_tab, ResizeHandleRow(equal_height=False):
                    with ExitStack() as stack:
                        if shared.opts.txt2img_settings_accordion:
                            stack.enter_context(gr.Accordion("Open for Settings", open=False))
                        stack.enter_context(gr.Column(variant='compact', elem_id=f"{id_part}_settings"))

                        wildcard_txt2img.scripts_gallery.prepare_ui()

                        for category in ordered_ui_categories():
                            if category == "prompt":
                                toprow.create_inline_toprow_prompts()

                            elif category == "dimensions":
                                with FormRow():
                                    with gr.Column(elem_id=f"{id_part}_column_size", scale=4):
                                        width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id=f"{id_part}_width")
                                        height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id=f"{id_part}_height")

                                    with gr.Column(elem_id=f"{id_part}_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                        res_switch_btn = ToolButton(value=switch_values_symbol, elem_id=f"{id_part}_res_switch_btn", tooltip="Switch width/height")

                                    if opts.dimensions_and_batch_together:
                                        with gr.Column(elem_id=f"{id_part}_column_batch"):
                                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{id_part}_batch_count")
                                            batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{id_part}_batch_size")

                            elif category == "cfg":
                                with gr.Row():
                                    distilled_cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Distilled CFG Scale', value=3.5, elem_id=f"{id_part}_distilled_cfg_scale")
                                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='CFG Scale', value=7.0, elem_id=f"{id_part}_cfg_scale")
                                    #cfg_scale.change(lambda x: gr.update(interactive=(x != 1)), inputs=[cfg_scale], outputs=[toprow.negative_prompt], queue=False, show_progress=False)

                            elif category == "checkboxes":
                                with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                    pass

                            elif category == "accordions":
                                with gr.Row(elem_id=f"{id_part}_accordions", elem_classes="accordions"):
                                    with InputAccordion(False, label="Hires. fix", elem_id=f"{id_part}_hr") as enable_hr:
                                        with enable_hr.extra():
                                            hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution")

                                        with FormRow(elem_id=f"{id_part}_hires_fix_row1", variant="compact"):
                                            hr_upscaler = gr.Dropdown(label="Upscaler", elem_id=f"{id_part}_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                            hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id=f"{id_part}_hires_steps")
                                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id=f"{id_part}_denoising_strength")

                                        with FormRow(elem_id=f"{id_part}_hires_fix_row2", variant="compact"):
                                            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id=f"{id_part}_hr_scale")
                                            hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id=f"{id_part}_hr_resize_x")
                                            hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id=f"{id_part}_hr_resize_y")

                                        with FormRow(elem_id=f"{id_part}_hires_fix_row_cfg", variant="compact"):
                                            hr_distilled_cfg = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label="Hires Distilled CFG Scale", value=3.5, elem_id=f"{id_part}_hr_distilled_cfg")
                                            hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label="Hires CFG Scale", value=7.0, elem_id=f"{id_part}_hr_cfg")

                                        with FormRow(elem_id=f"{id_part}_hires_fix_row3", variant="compact", visible=shared.opts.hires_fix_show_sampler) as hr_checkpoint_container:
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

                                        with FormRow(elem_id=f"{id_part}_hires_fix_row3b", variant="compact", visible=shared.opts.hires_fix_show_sampler) as hr_sampler_container:
                                            hr_sampler_name = gr.Dropdown(label='Hires sampling method', elem_id="hr_sampler", choices=["Use same sampler"] + sd_samplers.visible_sampler_names(), value="Use same sampler")
                                            hr_scheduler = gr.Dropdown(label='Hires schedule type', elem_id="hr_scheduler", choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers], value="Use same scheduler")

                                        with FormRow(elem_id=f"{id_part}_hires_fix_row4", variant="compact", visible=shared.opts.hires_fix_show_prompts) as hr_prompts_container:
                                            with gr.Column():
                                                hr_prompt = gr.Textbox(label="Hires prompt", elem_id="hires_prompt", show_label=False, lines=3, placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", elem_classes=["prompt"])
                                            with gr.Column():
                                                hr_negative_prompt = gr.Textbox(label="Hires negative prompt", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", elem_classes=["prompt"])

                                        hr_cfg.change(lambda x: gr.update(interactive=(x != 1)), inputs=[hr_cfg], outputs=[hr_negative_prompt], queue=False, show_progress=False)

                                    wildcard_txt2img.scripts_gallery.setup_ui_for_section(category)

                            elif category == "batch":
                                if not opts.dimensions_and_batch_together:
                                    with FormRow(elem_id=f"{id_part}_column_batch"):
                                        batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{id_part}_batch_count")
                                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{id_part}_batch_size")

                            elif category == "override_settings":
                                with FormRow(elem_id=f"{id_part}_override_settings_row") as row:
                                    override_settings = create_override_settings_dropdown('wg', row)

                            elif category == "scripts":
                                with FormGroup(elem_id=f"{id_part}_script_container"):
                                    custom_inputs = wildcard_txt2img.scripts_gallery.setup_ui()

                            if category not in {"accordions"}:
                                wildcard_txt2img.scripts_gallery.setup_ui_for_section(category)

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
                        

                extra_networks_ui = ui_extra_networks.create_ui(blocks, [prompt_generation_tab], f"{id_part}")
                ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)

                extra_tabs.__exit__()
            


    steps = wildcard_txt2img.scripts_gallery.script('Sampler').steps
    sampler_name = wildcard_txt2img.scripts_gallery.script('Sampler').sampler_name
    scheduler = wildcard_txt2img.scripts_gallery.script('Sampler').scheduler
    
    seed = wildcard_txt2img.scripts_gallery.script('Seed').seed

    gallery_change_args = dict(
        fn=change_gallery_selection,
        inputs=[gallery_selection],
        outputs=[gallery_name, toprow.prompt, toprow.negative_prompt, steps, sampler_name, scheduler, seed, width, height, batch_count, batch_size, cfg_scale, distilled_cfg_scale, output_panel.gallery, output_panel.generation_info, output_panel.infotext]
    )

    # element functions
    gallery_selection.change(**gallery_change_args)
    refresh_galleries.click(
        fn=click_refresh_galleries,
        inputs=[],
        outputs=[gallery_selection]
    )
    add_gallery.click(
        fn=click_add_gallery,
        inputs=[gallery_name],
        outputs=[gallery_selection, gallery_selection]
    ).then(**gallery_change_args)
    rename_gallery.click(
        fn=click_rename_gallery,
        inputs=[gallery_name],
        outputs=[gallery_selection, gallery_selection]
    ).then(**gallery_change_args)
    width.change(
        fn=change_width,
        inputs=[width]
    )
    height.change(
        fn=change_height,
        inputs=[height]
    )
    res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

    batch_count.change(
        fn=change_batch_count,
        inputs=[batch_count]
    )
    batch_size.change(
        fn=change_batch_size,
        inputs=[batch_size]
    )
    distilled_cfg_scale.change(
        fn=change_distilled_cfg_scale,
        inputs=[distilled_cfg_scale]
    )
    cfg_scale.change(
        fn=change_cfg_scale,
        inputs=[cfg_scale], 
        outputs=[toprow.negative_prompt]
    )
    toprow.prompt.change(
        fn=change_prompt,
        inputs=[toprow.prompt]
    )
    toprow.negative_prompt.change(
        fn=change_negative_prompt,
        inputs=[toprow.negative_prompt]
    )
    steps.change(
        fn=change_steps,
        inputs=[steps],
    )
    sampler_name.change(
        fn=change_sampler_name,
        inputs=[sampler_name],
    )
    scheduler.change(
        fn=change_scheduler,
        inputs=[scheduler],
    )
    seed.change(
        fn=change_seed,
        inputs=[seed],
    )

    txt2img_inputs = [
        dummy_component,
        toprow.prompt,
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

    txt2img_args = dict(
        #modules.txt2img
        fn=wrap_gradio_gpu_call(wildcard_txt2img.txt2img_prompt, extra_outputs=[None, '', '']),
        _js="wg_gal_submit",
        inputs=txt2img_inputs,
        outputs=txt2img_outputs,
        show_progress=False,
    )

    toprow.prompt.submit(**txt2img_args)
    toprow.submit.click(**txt2img_args)

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