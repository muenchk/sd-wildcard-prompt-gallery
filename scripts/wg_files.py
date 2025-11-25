
try:
    from tkinter import filedialog, Tk
except ImportError:
    pass
import sys
import os

ENV_EXCLUSION = []

def get_folder_path(folder_path: str = "") -> str:
    """
    Opens a folder dialog to select a folder, allowing the user to navigate and choose a folder.
    If no folder is selected, returns the initially provided folder path or an empty string if not provided.
    This function is conditioned to skip the folder dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - folder_path (str): The initial folder path or an empty string by default. Used as the fallback if no folder is selected.

    Returns:
    - str: The path of the folder selected by the user, or the initial `folder_path` if no selection is made.

    Raises:
    - TypeError: If `folder_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the folder dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the folder dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter type
    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")

    try:
        # Check for environment variable conditions
        if any(var in os.environ for var in ENV_EXCLUSION) or sys.platform == "darwin":
            return folder_path or ""

        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory(initialdir=folder_path or ".")
        root.destroy()
        return selected_folder or folder_path
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}") from e

def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)

def get_any_file_path(file_path: str = "") -> str:
    """
    Opens a file dialog to select any file, allowing the user to navigate and choose a file.
    If no file is selected, returns the initially provided file path or an empty string if not provided.
    This function is conditioned to skip the file dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - file_path (str): The initial file path or an empty string by default. Used as the fallback if no file is selected.

    Returns:
    - str: The path of the file selected by the user, or the initial `file_path` if no selection is made.

    Raises:
    - TypeError: If `file_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the file dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the file dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter type
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    try:
        # Check for environment variable conditions
        if (
            not any(var in os.environ for var in ENV_EXCLUSION)
            and sys.platform != "darwin"
        ):
            current_file_path: str = file_path

            initial_dir, initial_file = get_dir_and_file(file_path)

            # Initialize a hidden Tkinter window for the file dialog
            root = Tk()
            root.wm_attributes("-topmost", 1)
            root.withdraw()

            try:
                # Open the file dialog and capture the selected file path
                file_path = filedialog.askopenfilename(
                    initialdir=initial_dir,
                    initialfile=initial_file
                )
            except Exception as e:
                raise RuntimeError(f"Failed to open file dialog: {e}")
            finally:
                root.destroy()

            # Fallback to the initial path if no selection is made
            if not file_path:
                file_path = current_file_path
    except KeyError as e:
        raise EnvironmentError(f"Failed to access environment variables: {e}")

    # Return the selected or fallback file path
    return file_path

from modules.ui_common import plaintext_to_html
from PIL import Image
import piexif
import piexif.helper
from modules.images import IGNORED_INFO_KEYS
import json
import modules.sd_samplers as sd_samplers
from modules import errors

def read_info_from_image(image: Image.Image) -> tuple[str | None, dict]:
    """Read generation info from an image, checking standard metadata first, then stealth info if needed."""

    def read_standard():
        items = (image.info or {}).copy()

        geninfo = items.pop('parameters', None)

        if "exif" in items:
            exif_data = items["exif"]
            try:
                exif = piexif.load(exif_data)
            except OSError:
                # memory / exif was not valid so piexif tried to read from a file
                exif = None
            exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
            try:
                exif_comment = piexif.helper.UserComment.load(exif_comment)
            except ValueError:
                exif_comment = exif_comment.decode('utf8', errors="ignore")

            if exif_comment:
                geninfo = exif_comment
        elif "comment" in items: # for gif
            if isinstance(items["comment"], bytes):
                geninfo = items["comment"].decode('utf8', errors="ignore")
            else:
                geninfo = items["comment"]

        for field in IGNORED_INFO_KEYS:
            items.pop(field, None)

        if items.get("Software", None) == "NovelAI":
            try:
                json_info = json.loads(items["Comment"])
                sampler = sd_samplers.samplers_map.get(json_info["sampler"], "Euler a")

                geninfo = f"""{items["Description"]}
    Negative prompt: {json_info["uc"]}
    Steps: {json_info["steps"]}, Sampler: {sampler}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
            except Exception:
                errors.report("Error parsing NovelAI image generation parameters", exc_info=True)

        return geninfo, items
    
    geninfo, items = read_standard()
    #if geninfo is None:
    #    geninfo = stealth_infotext.read_info_from_image_stealth(image)

    if geninfo is None:
        geninfo = ""
    return geninfo, items


def run_pnginfo(image):
    if image is None:
        return '', '', ''
    geninfo, items = read_info_from_image(image)
    items = {**{'parameters': geninfo}, **items}

    info = ''
    for key, text in items.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return '', geninfo, info