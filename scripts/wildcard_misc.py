from modules.ui_common import OutputPanel, save_files, plaintext_to_html
folder_symbol = '\U0001f4c2'  # 📂
from modules.ui_components import ToolButton
from modules import shared, util, ui_tempdir, call_queue
from modules.shared import opts, cmd_opts
import modules.infotext_utils as parameters_copypaste
from modules import processing, infotext_utils, images
import modules
import modules.scripts
import json
import scripts.prompt_gallery as prompt_gallery

import gradio as gr

import os
import os.path

import scripts.wildcard_txt2img as wildcard_txt2img
import scripts.wildcard_json as wildcard_json

import collections.abc

import random

from scripts.wildcard_data import get_gen_data, get_wildcard_manager, key_wildcard, key_prompt, key_prompt_second

def delete_old_images(imagelist):
    # delete the passed old files
    dir = wildcard_json.get_base_dir()
    if isinstance(imagelist, dict):
        #1234for obj in imagelist:
        for cap, obj in imagelist.items():
            #1234path = os.path.join(dir, obj[wildcard_json.key_ia_filename])
            path = os.path.join(dir, obj[wildcard_json.key_img])
            print(path)
            if (os.path.exists(path)):
                print("exists")
            try:
                os.remove(path)
                os.removedirs(os.path.dirname(path))
            except Exception as e:
                pass
    elif isinstance(imagelist, collections.abc.Iterable):
        for img in imagelist:
            path = os.path.join(dir, img[wildcard_json.key_img])
            print(path)
            if (os.path.exists(path)):
                print("exists")
            try:
                os.remove(path)
                os.removedirs(os.path.dirname(path))
            except Exception as e:
                pass
    else:
        print("[ERROR] [WG] imagelist for deletion not of known type")

def delete_gallery(path):
    for root,dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except:
                pass
        for dir in dirs:
            try:
                os.removedirs(os.path.join(root, dir))
            except Exception as e:
                print(e)
    try:
        os.removedirs(path)
    except Exception as e:
        print(e)
        


def save_images(p, processed, generation_info_js, gen_html, mode):
    # get image num and pathes
    num = len(processed.images)
    dir = wildcard_json.get_base_dir()
    wildcard = get_gen_data(key_wildcard)
    wildcard_path = wildcard.replace('/', os.path.sep)
    wildcard_path = wildcard_path.replace('\\', os.path.sep)
    image_path = os.path.join(dir, wildcard_path)
    card = wildcard_json.get_wildcard(wildcard)
    # delete old images if we are generating all combinations
    if mode == 1:
        delete_old_images(card[wildcard_json.key_image_array])
    # reset image list if we are generating all new images
    if mode == 1:
        card[wildcard_json.key_image_array] = {}
    # create directories
    try:
        os.makedirs(image_path)
    except Exception:
        pass

    to_delete = []

    # save the files and store their information
    for i in range(num):
        fullfn, txt_fullfn = images.save_image(processed.images[i][0], image_path, "", processed.all_seeds[i], processed.all_prompts[i], opts.samples_format, info=processed.infotext(p, i), p = p)
        imgobj = {}
        print("i: " + str(i) + "filename: " + fullfn[len(str(dir))+1:] + "caption: " + processed.images[i][1])
        imgobj[wildcard_json.key_ia_filename] = fullfn[len(str(dir))+1:]
        imgobj[wildcard_json.key_ia_caption] = processed.images[i][1]
        #1234card[wildcard_json.key_image_array].append(imgobj)
        if processed.images[i][1] in card[wildcard_json.key_image_array]:
            # if image exists, add it to the deletion
            to_delete.append(card[wildcard_json.key_image_array][processed.images[i][1]])

        image_details = {}
        image_details[wildcard_json.key_img] = fullfn[len(str(dir))+1:]
        image_details[wildcard_json.key_prompt] = processed.all_prompts[i]
        image_details[wildcard_json.key_negative_prompt] = processed.all_negative_prompts[i]
        image_details[wildcard_json.key_seed] = processed.all_seeds[i]
        image_details[wildcard_json.key_seed_subseed] = processed.all_subseeds[i]
        image_details[wildcard_json.key_seed_subseed_strength] = processed.subseed_strength
        image_details[wildcard_json.key_width] = processed.width
        image_details[wildcard_json.key_height] = processed.height
        image_details[wildcard_json.key_sampler] = processed.sampler_name
        image_details[wildcard_json.key_cfg] = processed.cfg_scale
        image_details[wildcard_json.key_sampling_steps] = processed.steps
        image_details[wildcard_json.key_batch_size] = processed.batch_size
        image_details[wildcard_json.key_restore_faces] = processed.restore_faces
        image_details[wildcard_json.key_face_restoration_model] = processed.face_restoration_model
        image_details[wildcard_json.key_sd_model_name] = processed.sd_model_name
        image_details[wildcard_json.key_sd_model_hash] = processed.sd_model_hash
        image_details[wildcard_json.key_sd_vae_name] = processed.sd_vae_name
        image_details[wildcard_json.key_sd_vae_hash] = processed.sd_vae_hash
        image_details[wildcard_json.key_seed_resize_from_w] = processed.seed_resize_from_w
        image_details[wildcard_json.key_seed_resize_from_h] = processed.seed_resize_from_h
        image_details[wildcard_json.key_denoising_strength] = processed.denoising_strength
        image_details[wildcard_json.key_extra_generation_params] = processed.extra_generation_params
        image_details[wildcard_json.key_index_of_first_image] = processed.index_of_first_image
        image_details[wildcard_json.key_infotexts] = processed.infotexts[i]
        image_details[wildcard_json.key_styles] = processed.styles
        image_details[wildcard_json.key_job_timestamp] = processed.job_timestamp
        image_details[wildcard_json.key_clip_skip] = processed.clip_skip
        image_details[wildcard_json.key_is_using_inpainting_conditioning] = processed.is_using_inpainting_conditioning
        image_details[wildcard_json.key_version] = processed.version
        card[wildcard_json.key_image_array][processed.images[i][1]] = image_details

    wildcard_json.update_wildcard(wildcard, wildcard_json.key_image_array, card[wildcard_json.key_image_array])

    if mode == 3 and len(to_delete) > 0:
        delete_old_images(to_delete)
    #wildcard_json.update_wildcard(wildcard, wildcard_json.key_image_generation_info, generation_info_js)
    #wildcard_json.update_wildcard(wildcard, wildcard_json.key_image_generation_html, gen_html)

    wildcard_json.write_to_config()


def save_image_to_gallery(image, generation_info_s, img_index):
    generation_info = json.loads(generation_info_s)
    if img_index < 0 or img_index >= len(generation_info[wildcard_json.key_infotexts]) or "loaded" in generation_info:
        return # if loaded in generation_info, we loaded the images from the databse, i.e. the image is already in the gallery
    return save_image_to_gallery_intern(image[img_index], generation_info_s, img_index)
def save_image_to_gallery_intern(img, generation_info, img_index):
    generation_info = json.loads(generation_info)
    dir = wildcard_json.get_base_dir() / "galleries"
    gallery_name = prompt_gallery.get_gallery_name()
    if (gallery_name == ""):
        return
    image_path = os.path.join(dir, gallery_name)
    gallery = wildcard_json.get_gallery(gallery_name)

    to_delete = []

    fullfn, txt_fullfn = images.save_image(img[0] if isinstance(img, tuple) else img, image_path, "", generation_info[wildcard_json.key_all_seeds][img_index], generation_info[wildcard_json.key_all_prompts][img_index], opts.samples_format, info=generation_info[wildcard_json.key_infotexts][img_index])
    imgobj={}
    caption = img[1] if isinstance(img, tuple) else "img_" + str(random.randint(0, 9223372036854775806))
    # check if caption already exists and add a random number to the end until it doesn't
    if caption in gallery[wildcard_json.key_image_array]:
        caption = caption + "_" + str(random.randint(0, 9223372036854775806))

    imgobj[wildcard_json.key_ia_filename] = fullfn[len(str(dir))+1:]
    imgobj[wildcard_json.key_ia_caption] = caption
    if caption in gallery[wildcard_json.key_image_array]:
        to_delete.append(gallery[wildcard_json.key_image_array][caption])
    
    image_details = {}
    image_details[wildcard_json.key_img] = fullfn[len(str(dir))+1:]
    image_details[wildcard_json.key_prompt] = generation_info[wildcard_json.key_all_prompts][img_index]
    image_details[wildcard_json.key_negative_prompt] = generation_info[wildcard_json.key_all_negative_prompts][img_index]
    image_details[wildcard_json.key_seed] = generation_info[wildcard_json.key_all_seeds][img_index]
    image_details[wildcard_json.key_seed_subseed] = generation_info[wildcard_json.key_seed_all_subseeds][img_index]
    image_details[wildcard_json.key_seed_subseed_strength] = generation_info[wildcard_json.key_seed_subseed_strength]
    image_details[wildcard_json.key_width] = generation_info[wildcard_json.key_width]
    image_details[wildcard_json.key_height] = generation_info[wildcard_json.key_height]
    image_details[wildcard_json.key_sampler] = generation_info[wildcard_json.key_sampler]
    if wildcard_json.key_scheduler in generation_info:
        image_details[wildcard_json.key_scheduler] = generation_info[wildcard_json.key_scheduler]
    image_details[wildcard_json.key_cfg] = generation_info[wildcard_json.key_cfg]
    image_details[wildcard_json.key_sampling_steps] = generation_info[wildcard_json.key_sampling_steps]
    image_details[wildcard_json.key_batch_size] = generation_info[wildcard_json.key_batch_size]
    image_details[wildcard_json.key_restore_faces] = generation_info[wildcard_json.key_restore_faces]
    image_details[wildcard_json.key_face_restoration_model] = generation_info[wildcard_json.key_face_restoration_model]
    image_details[wildcard_json.key_sd_model_name] = generation_info[wildcard_json.key_sd_model_name]
    image_details[wildcard_json.key_sd_model_hash] = generation_info[wildcard_json.key_sd_model_hash]
    image_details[wildcard_json.key_sd_vae_name] = generation_info[wildcard_json.key_sd_vae_name]
    image_details[wildcard_json.key_sd_vae_hash] = generation_info[wildcard_json.key_sd_vae_hash]
    image_details[wildcard_json.key_seed_resize_from_w] = generation_info[wildcard_json.key_seed_resize_from_w]
    image_details[wildcard_json.key_seed_resize_from_h] = generation_info[wildcard_json.key_seed_resize_from_h]
    image_details[wildcard_json.key_denoising_strength] = generation_info[wildcard_json.key_denoising_strength]
    image_details[wildcard_json.key_extra_generation_params] = generation_info[wildcard_json.key_extra_generation_params]
    image_details[wildcard_json.key_index_of_first_image] = generation_info[wildcard_json.key_index_of_first_image]
    image_details[wildcard_json.key_infotexts] = generation_info[wildcard_json.key_infotexts][img_index]
    image_details[wildcard_json.key_styles] = generation_info[wildcard_json.key_styles]
    image_details[wildcard_json.key_job_timestamp] = generation_info[wildcard_json.key_job_timestamp]
    image_details[wildcard_json.key_clip_skip] = generation_info[wildcard_json.key_clip_skip]
    image_details[wildcard_json.key_is_using_inpainting_conditioning] = generation_info[wildcard_json.key_is_using_inpainting_conditioning]
    image_details[wildcard_json.key_version] = generation_info[wildcard_json.key_version]
    gallery[wildcard_json.key_image_array][caption] = image_details

    
    wildcard_json.update_gallery(gallery_name, wildcard_json.key_image_array, gallery[wildcard_json.key_image_array])

    delete_old_images(to_delete)

    wildcard_json.write_to_config()
    
def remove_image_from_gallery(image, generation_info, html_info, img_index):
    generation_info = json.loads(generation_info)
    if img_index < 0 or img_index >= len(generation_info[wildcard_json.key_infotexts]) or "loaded" not in generation_info:
        return # if loaded in generation_info, we loaded the images from the databse, i.e. the image is already in the gallery
    dir = wildcard_json.get_base_dir() / "galleries"
    gallery_name = prompt_gallery.get_gallery_name()
    if (gallery_name == ""):
        return
    image_path = os.path.join(dir, gallery_name)
    gallery = wildcard_json.get_gallery(gallery_name)

    to_delete = [gallery[wildcard_json.key_image_array][image[img_index][1]]]

    delete_old_images(to_delete)
    del gallery[wildcard_json.key_image_array][image[img_index][1]]

    del image[img_index]
    image_generation_info, image_generation_html = get_initial_generation_info(gallery[wildcard_json.key_image_array])

    return image, image_generation_info, image_generation_html
    

def get_initial_generation_info(imagelist):
    retobj = {}
    retobj[wildcard_json.key_all_prompts] = []
    retobj[wildcard_json.key_all_negative_prompts] = []
    retobj[wildcard_json.key_all_seeds] = []
    retobj[wildcard_json.key_seed_all_subseeds] = []
    retobj[wildcard_json.key_infotexts] = []
    retobj["loaded"] = True

    for cap, obj in imagelist.items():
        for key, val in obj.items():
            if key == wildcard_json.key_prompt:
                retobj[wildcard_json.key_all_prompts].append(val)
                retobj[key] = val
            if key == wildcard_json.key_negative_prompt:
                retobj[wildcard_json.key_all_negative_prompts].append(val)
                retobj[key] = val
            if key == wildcard_json.key_seed:
                retobj[wildcard_json.key_all_seeds].append(val)
                retobj[key] = val
            if key == wildcard_json.key_seed_subseed:
                retobj[wildcard_json.key_seed_all_subseeds].append(val)
                retobj[key] = val
            if key == wildcard_json.key_infotexts:
                retobj[wildcard_json.key_infotexts].append(val)
            else:
                retobj[key] = val
    retobj[wildcard_json.key_index_of_first_image] = 0
    
    return json.dumps(retobj, default=lambda o: None), plaintext_to_html(retobj[wildcard_json.key_infotexts][0] if len(imagelist) > 0 else "")
        

def update_generation_info(generation_info, html_info, img_index):
    try:
        generation_info = json.loads(generation_info)
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info, gr.update()
        return plaintext_to_html(generation_info["infotexts"][img_index]), gr.update()
    except Exception:
        pass
    # if the json parse or anything else fails, just return the old html_info
    return html_info, gr.update()

def create_wg_output_panel(tabname, outdir):
    output_panel = OutputPanel()
    def open_folder(f, images=None, index=None):
        if shared.cmd_opts.hide_ui_dir_config:
            return

        try:
            if 'Sub' in shared.opts.open_dir_button_choice:
                image_dir = os.path.split(images[index]["name"].rsplit('?', 1)[0])[0]
                if 'temp' in shared.opts.open_dir_button_choice or not ui_tempdir.is_gradio_temp_path(image_dir):
                    f = image_dir
        except Exception:
            pass

        util.open_folder(f)
    with gr.Group(elem_id=f"{tabname}_gallery_container"):
        output_panel.gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery", columns=4, preview=False, height="max-content", interactive=False, type="pil", object_fit="contain")
    with gr.Row(elem_id=f"image_buttons_wg", elem_classes="image-buttons"):
        open_folder_button = ToolButton(folder_symbol, elem_id=f'{tabname}_open_folder', visible=not shared.cmd_opts.hide_ui_dir_config, tooltip="Open images output directory.")

        save = ToolButton('💾', elem_id=f'save_{tabname}', tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).")
        save_zip = ToolButton('🗃️', elem_id=f'save_zip_{tabname}', tooltip=f"Save zip archive with images to a dedicated directory ({shared.opts.outdir_save})")

        buttons = {
            'img2img': ToolButton('🖼️', elem_id=f'{tabname}_send_to_img2img', tooltip="Send image and generation parameters to img2img tab."),
            'inpaint': ToolButton('🎨️', elem_id=f'{tabname}_send_to_inpaint', tooltip="Send image and generation parameters to img2img inpaint tab."),
            'extras': ToolButton('📐', elem_id=f'{tabname}_send_to_extras', tooltip="Send image and generation parameters to extras tab.")
        }

        send_to_txt2img = gr.Button("send to txt2img", elem_id=f'{tabname}_send_to_txt2img', tooltip="Send image and generation parameters to txt2img tab.")
        if tabname == "wg_prompt":
            add_to_gallery = gr.Button("Add to gallery", elem_id=f"{tabname}_add_to_gallery", tooltip="Add the image to the gallery")
            remove_from_gallery = gr.Button("Remove from Gallery", elem_id=f"{tabname}_remove_from_gallery", tooltip="Remove image from image gallery")
            

        
    open_folder_button.click(
        fn=lambda images, index: open_folder(shared.opts.outdir_samples or outdir, images, index),
        _js="(y, w) => [y, selected_gallery_index()]",
        inputs=[
            output_panel.gallery,
            open_folder_button,  # placeholder for index
        ],
        outputs=[],
    )
    download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_{tabname}')

    with gr.Group():
        output_panel.infotext = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext")
        output_panel.html_log = gr.HTML(elem_id=f'html_log_{tabname}', elem_classes="html-log")

        output_panel.generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
        generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")
        generation_info_button.click(
            fn=update_generation_info,
            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
            inputs=[output_panel.generation_info, output_panel.infotext, output_panel.infotext],
            outputs=[output_panel.infotext, output_panel.infotext],
            show_progress=False,
        )

        save.click(
            fn=call_queue.wrap_gradio_call_no_job(save_files),
            _js="(x, y, z, w) => [x, y, false, selected_gallery_index()]",
            inputs=[
                output_panel.generation_info,
                output_panel.gallery,
                output_panel.infotext,
                output_panel.infotext,
            ],
            outputs=[
                download_files,
                output_panel.html_log,
            ],
            show_progress=False,
        )

        save_zip.click(
            fn=call_queue.wrap_gradio_call_no_job(save_files),
            _js="(x, y, z, w) => [x, y, true, selected_gallery_index()]",
            inputs=[
                output_panel.generation_info,
                output_panel.gallery,
                output_panel.infotext,
                output_panel.infotext,
            ],
            outputs=[
                download_files,
                output_panel.html_log,
            ]
        )

        if tabname == "wg_prompt":
            add_to_gallery.click(
                fn=save_image_to_gallery,
                _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                inputs=[output_panel.gallery, output_panel.generation_info, output_panel.infotext],
                show_progress=False
            )
            remove_from_gallery.click(
                fn=remove_image_from_gallery,
                _js="function(w, x, y, z){ return [w, x, y, selected_gallery_index()] }",
                inputs=[output_panel.gallery, output_panel.generation_info, output_panel.infotext, output_panel.infotext],
                outputs=[output_panel.gallery, output_panel.generation_info, output_panel.infotext],
                show_progress=False
            )
    
    paste_field_names_txt = modules.scripts.scripts_txt2img.paste_field_names
    paste_field_names_img = modules.scripts.scripts_img2img.paste_field_names

    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
            paste_button=send_to_txt2img, tabname='txt2img', source_image_component=output_panel.gallery,source_text_component=output_panel.generation_info
        ))

    for paste_tabname, paste_button in buttons.items():
        parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
            paste_button=paste_button, tabname=paste_tabname, source_image_component=output_panel.gallery,source_text_component=output_panel.generation_info,
            paste_field_names=paste_field_names_img
        ))

    return output_panel