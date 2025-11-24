
from contextlib import closing

from gradio.components import Checkbox
from gradio.routes import BackgroundTask
import modules.scripts
from modules_forge import main_thread
from modules.ui import plaintext_to_html
import modules.shared as shared
from modules.shared import opts
import modules.txt2img 
from collections.abc import Sequence
from modules import processing, infotext_utils, images
from modules.processing import fix_seed
import gradio as gr
import math
from itertools import cycle, islice, product
import os
import collections.abc

from PIL import Image, ImageFont, ImageDraw, ImageOps

from modules.paths_internal import roboto_ttf_file

from scripts.wildcard_data import get_gen_data, get_wildcard_manager, key_wildcard, key_prompt, key_prompt_second
import scripts.wildcard_json as wildcard_json
import scripts.wildcard_misc as wildcard_misc

from dynamicprompts.parser.parse import ParserConfig
from dynamicprompts.generators.promptgenerator import GeneratorException
from dynamicprompts.wildcards import WildcardManager
from dynamicprompts.generators import (
    BatchedCombinatorialPromptGenerator,
    CombinatorialPromptGenerator,
    DummyGenerator,
    FeelingLuckyGenerator,
    JinjaGenerator,
    PromptGenerator,
    RandomPromptGenerator,
)

def get_seeds(
    p,
    num_seeds,
    use_fixed_seed,
    is_combinatorial=False,
    combinatorial_batches=1
    ) -> tuple[list[int], list[int]]:
    if p.subseed_strength != 0:
        seed = int(p.all_seeds[0])
        subseed = int(p.all_subseeds[0])
    else:
        seed = int(p.seed)
        subseed = int(p.subseed)

    if use_fixed_seed:
        if is_combinatorial:
            all_seeds = []
            all_subseeds = [subseed] * num_seeds
            for i in range(combinatorial_batches):
                all_seeds.extend([seed + i] * (num_seeds // combinatorial_batches))
        else:
            all_seeds = [seed] * num_seeds
            all_subseeds = [subseed] * num_seeds
    else:
        if p.subseed_strength == 0:
            all_seeds = [seed + i for i in range(num_seeds)]
        else:
            all_seeds = [seed] * num_seeds

        all_subseeds = [subseed + i for i in range(num_seeds)]

    return all_seeds, all_subseeds


def generate_prompts(
    prompt_generator: PromptGenerator,
    negative_prompt_generator: PromptGenerator,
    prompt: str,
    negative_prompt: str | None,
    num_prompts: int,
    seeds: list[int] | None,
) -> tuple[list[str], list[str]]:
    """
    Generate positive and negative prompts.

    Parameters:
    - prompt_generator: Object that generates positive prompts.
    - negative_prompt_generator: Object that generates negative prompts.
    - prompt: Base text for positive prompts.
    - negative_prompt: Base text for negative prompts.
    - num_prompts: Number of prompts to generate.
    - seeds: List of seeds for prompt generation.

    Returns:
    - Tuple containing list of positive and negative prompts.
    """
    all_prompts = prompt_generator.generate(prompt, num_prompts, seeds=seeds) or [""]
    print("all prompts")
    print (all_prompts)

    negative_seeds = seeds if negative_prompt else None

    all_negative_prompts = negative_prompt_generator.generate(
        negative_prompt,
        num_prompts,
        seeds=negative_seeds,
    ) or [""]

    if num_prompts is None:
        return generate_prompt_cross_product(all_prompts, all_negative_prompts)

    return all_prompts, repeat_iterable_to_length(all_negative_prompts, num_prompts)


def generate_prompt_cross_product(
    prompts: list[str],
    negative_prompts: list[str],
) -> tuple[list[str], list[str]]:
    """
    Create a cross product of all the items in `prompts` and `negative_prompts`.
    Return the positive prompts and negative prompts in two separate lists

    Parameters:
    - prompts: List of prompts
    - negative_prompts: List of negative prompts

    Returns:
    - Tuple containing list of positive and negative prompts
    """
    if not (prompts and negative_prompts):
        return [], []

    # noqa to remain compatible with python 3.9, see issue #601
    new_positive_prompts, new_negative_prompts = zip(
        *product(prompts, negative_prompts),  # noqa: B905
    )
    print("new positive prompts")
    print (new_positive_prompts)
    return list(new_positive_prompts), list(new_negative_prompts)


def repeat_iterable_to_length(iterable, length: int) -> list:
    """Repeat an iterable to a given length.

    If the iterable is shorter than the desired length, it will be repeated
    until it is long enough. If it is longer than the desired length, it will
    be truncated.

    Args:
        iterable (Iterable): The iterable to repeat.
        length (int): The desired length of the iterable.

    Returns:
        list: The repeated iterable.

    """
    return list(islice(cycle(iterable), length))


def update_prompts(p, mode):
    parser_config = ParserConfig(
        variant_start=opts.dp_parser_variant_start,
        variant_end=opts.dp_parser_variant_end,
        wildcard_wrap=opts.dp_parser_wildcard_wrap,
    )

    fix_seed(p)

    print(p.prompt)
    original_prompt = p.all_prompts[0] if p.all_prompts else p.prompt
    original_negative_prompt = p.all_negative_prompts[0] if p.all_negative_prompts else p.negative_prompt

    original_seed = p.seed
    num_images = p.n_iter * p.batch_size

    # combinatorial
    if mode == 1:
        num_images = None

    try:
        if (mode == 1):
            generator = CombinatorialPromptGenerator(
                get_wildcard_manager(),
                parser_config=parser_config,
                ignore_whitespace=False
            )

            negative_generator = CombinatorialPromptGenerator(
                get_wildcard_manager(),
                parser_config=parser_config,
                ignore_whitespace=False
            )
        else:
            generator = RandomPromptGenerator(
                get_wildcard_manager(),
                seed=original_seed,
                parser_config=parser_config,
                unlink_seed_from_prompt=False,
                ignore_whitespace=opts.dp_ignore_whitespace
            )

            negative_generator = RandomPromptGenerator(
                get_wildcard_manager(),
                seed=original_seed,
                parser_config=parser_config,
                unlink_seed_from_prompt=False,
                ignore_whitespace=opts.dp_ignore_whitespace
            )

        all_seeds = None
        if num_images:
            p.all_seeds, p.all_subseeds = get_seeds(
                p, num_images,
                False,
                True,
                1
            )
            all_seeds = p.all_seeds

        
        all_prompts, all_negative_prompts = generate_prompts(
            prompt_generator=generator,
            negative_prompt_generator=negative_generator,
            prompt=original_prompt,
            negative_prompt=original_negative_prompt,
            num_prompts=num_images,
            seeds=all_seeds,
        )
    except GeneratorException as e:
        print(e)
        all_prompts = [str(e)]
        all_negative_prompts = [str(e)]

    updated_count = len(all_prompts)
    p.n_iter = math.ceil(updated_count / p.batch_size)

    if num_images != updated_count:
        p.all_seeds, p.all_subseeds = get_seeds(
            p,
            updated_count,
            False,
            mode == 1,
            1,
        )
    if (updated_count > 1):
        print(f"Wildcard Gallery: Prompt matrix will ceate {updated_count} images in a total of {p.n_iter} batches.")

    for i in range(len(all_prompts)):
        #if all_prompts[i] is list or all_prompts[i] is tuple:
        if isinstance(all_prompts[i], Sequence):
            tprompt = ""
            for string in all_prompts[i]:
                tprompt += string
            all_prompts[i] = tprompt
            
    p.seeds = p.all_seeds
    p.all_prompts = all_prompts
    p.prompt = p.all_prompts
    p.all_negative_prompts = all_negative_prompts
    p.negative_prompt = p.all_negative_prompts
    p.prompt_for_display = original_prompt

def write_on_image(img, msg):
    ix,iy = img.size
    draw = ImageDraw.Draw(img)
    margin=2
    fontsize=get_gen_data(wildcard_json.key_fontsize)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(roboto_ttf_file, fontsize)
    text_height=iy-60
    tx = draw.textbbox((0,0),msg,font)
    draw.text((int((ix-tx[2])/2),text_height+margin),msg,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2),text_height-margin),msg,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2+margin),text_height),msg,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2-margin),text_height),msg,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2),text_height), msg,(222,222,222),font=font)
    return img

def write_images(p, processed):
    prompt = get_gen_data(key_prompt)
    prompt_rest = get_gen_data(key_prompt_second)
    print("prompt:  " + prompt)
    print("promtp2: " + prompt_rest)

    print ("prompt array length: " + str(len(processed.all_prompts)))
    print ("prompt seed length: " + str(len(processed.all_seeds)))
    print ("prompt image length: " + str(len(processed.images)))
    print ("prompt extra image length: " + str(len(processed.extra_images)))

    for i in range(len(processed.all_prompts)):
        tprompt = ""
        if processed.all_prompts[i] is list or processed.all_prompts[i] is tuple:
            for string in processed.all_prompts[i]:
                tprompt += string
            print("list: " + tprompt)
        else:
            tprompt = processed.all_prompts[i]
            print("str: " + tprompt)
        desc = tprompt[len(prompt):]
        desc = desc[:len(desc) - len(prompt_rest)]
        processed.images[i] = write_on_image(processed.images[i], desc), desc
        print ("original: " + processed.all_prompts[i] + " text: " + desc)
        if get_gen_data(wildcard_json.key_write_to_image):
            images.save_image(processed.images[i][0], p.outpath_samples, "", processed.all_seeds[i], processed.all_prompts[i], opts.samples_format, info=processed.infotext(p, i), p=p)

from modules.scripts import AlwaysVisible

scripts_custom:modules.scripts.ScriptRunner = None

scripts_gallery:modules.scripts.ScriptRunner = None

scripts_data = []
postprocessing_scripts_data = []

from modules import errors
def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)

    return default

class ScriptRunnerCust(modules.scripts.ScriptRunner):
    def __init__(self):
        super().__init__()
        self.disabled_scripts:list[str] = []

    def process(self, p):
        for script in self.ordered_scripts('process'):
            found:bool = False
            for dis in self.disabled_scripts:
                if dis in script.filename:
                    found=True
                    break
            if found == False:
            #if "dynamic_prompting" not in script.filename:
                try:
                    script_args = p.script_args[script.args_from:script.args_to]
                    script.process(p, *script_args)
                except Exception:
                    errors.report(f"Error running process: {script.filename}", exc_info=True)
                    
    def add_disabled_script(self, script:str):
        self.disabled_scripts.append(script)

def load_scripts():
    global scripts_data
    scripts_data.clear()
    for data in modules.scripts.scripts_data:
        script_class, path, basedir, module = data
        if "dynamic_prompting" not in path:
            scripts_data.append(data)
            

    global scripts_custom
    global scripts_gallery

    #print(scripts_data)
    scripts_custom = ScriptRunnerCust()
    scripts_custom.add_disabled_script("dynamic_prompting")

    scripts_gallery = ScriptRunnerCust()
    #scripts_custom = modules.scripts.ScriptRunner()

from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters

def txt2img_create_processing(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_additional_modules: list, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, hr_cfg: float, hr_distilled_cfg: float, override_settings_texts, *args, force_enable_hr=False):
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    p = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_additional_modules=hr_additional_modules,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == 'Use same scheduler' else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        hr_cfg=hr_cfg,
        hr_distilled_cfg=hr_distilled_cfg,
        override_settings=override_settings,
    )

    p.scripts = scripts_custom
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return p


def txt2img_function(id_task: str, request: gr.Request, mode:int, *args):
    """
    This function does the actual top-level txt2img work.
    It adapts running parameters to our needs, takes care of storing images, and
    getting individual prompts from the wildcard prompt.

    It can be operated in three modi yielding different results.

    Parameters:
        mode (int): The mode of operation of function determining what images are generated, how many and whether they are saved, and how they are saved. 
            [1] txt2img with all prompt combinations as result. images are saved and are added to the database, after all older images are deleted.
            [2] txt2img with random prompts and [batch_count] images. images are NOT saved and DO NOT override existing images.
            [3] txt2img with random prompts and [batch_count] images. images ARE saved and DO override existing images. existing images are NOT cleared prior to this operation.
    
    Returns:
        images: images that have been generated
        json: json containing image generation info
        info: info in html form
        comments: comments in html form
    """

    print("step 1")

    p = txt2img_create_processing(id_task, request, *args)

    p.steps = get_gen_data(wildcard_json.key_sampling_steps)
    p.sampler_name = get_gen_data(wildcard_json.key_sampler)
    p.scheduler = get_gen_data(wildcard_json.key_scheduler)
    p.seed = get_gen_data(wildcard_json.key_seed)
    p.do_not_save_grid = True

    print("step 2")

    with closing(p):
        #processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)
        processed = scripts_custom.run(p, *p.script_args)

        print("step 3")

        # resolve the wildcards
        update_prompts(p, mode)

        print("step 4")

        for prompt in p.all_prompts:
            print(prompt)
        
        if processed is None:
            processed = processing.process_images(p)

        print("step 5")

        # write on images
        write_images(p, processed)

    print("step 6")

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    gen_html = plaintext_to_html(processed.info)

    if (mode != 2):
        wildcard_misc.save_images(p, processed, generation_info_js, gen_html, mode)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, gen_html, plaintext_to_html(processed.comments, classname="comments")

def txt2img(id_task: str, request: gr.Request, *args):
    print("generate")
    wildcard_json.writeback_wildcard_changes()
    return main_thread.run_and_wait_result(txt2img_function, id_task, request, 1, *args)

def txt2img_samples(id_task: str, request: gr.Request, *args):
    print("samples")
    wildcard_json.writeback_wildcard_changes()
    return main_thread.run_and_wait_result(txt2img_function, id_task, request, 2, *args)

def txt2img_samples_save(id_task:str, request: gr.Request, *args):
    print("samples save")
    wildcard_json.writeback_wildcard_changes()
    #return main_thread.run_and_wait_result(txt2img_function, id_task, request, 3, *args)
    return txt2img_function(id_task, request, 3, *args)


    

def txt2img_function_prompt(id_task: str, request: gr.Request, *args):
    """
    This function does the actual top-level txt2img work.
    It adapts running parameters to our needs, takes care of storing images, and
    getting individual prompts from the wildcard prompt.

    It can be operated in three modi yielding different results.

    Parameters:
        mode (int): The mode of operation of function determining what images are generated, how many and whether they are saved, and how they are saved. 
            [1] txt2img with all prompt combinations as result. images are saved and are added to the database, after all older images are deleted.
            [2] txt2img with random prompts and [batch_count] images. images are NOT saved and DO NOT override existing images.
            [3] txt2img with random prompts and [batch_count] images. images ARE saved and DO override existing images. existing images are NOT cleared prior to this operation.
    
    Returns:
        images: images that have been generated
        json: json containing image generation info
        info: info in html form
        comments: comments in html form
    """

    p = txt2img_create_processing(id_task, request, *args)

    p.steps = get_gen_data(wildcard_json.key_sampling_steps)
    p.sampler_name = get_gen_data(wildcard_json.key_sampler)
    p.scheduler = get_gen_data(wildcard_json.key_scheduler)
    p.seed = get_gen_data(wildcard_json.key_seed)
    p.do_not_save_grid = True

    with closing(p):
        #processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)
        processed = scripts_custom.run(p, *p.script_args)

        # resolve the wildcards
        update_prompts(p, mode)

        for prompt in p.all_prompts:
            print(prompt)
        
        if processed is None:
            processed = processing.process_images(p)

        # write on images
        write_images(p, processed)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    gen_html = plaintext_to_html(processed.info)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, gen_html, plaintext_to_html(processed.comments, classname="comments")

def txt2img_prompt(id_task: str, request: gr.Request, *args):
    wildcard_json.writeback_gallery_changes()
    print("txt2img prompt version")
    return main_thread.run_and_wait_result(txt2img_function_prompt, id_task, request, *args)