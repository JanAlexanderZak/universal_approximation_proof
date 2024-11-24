import glob
import re

from PIL import Image


def create_gif(
    frame_folder: str,
    save_path: str,
    duration: int = 100,
):
    files = glob.glob(f"{frame_folder}/*.png")
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    frames = [Image.open(image) for image in files]
    frame_one = frames[0]

    frame_one.save(
        save_path,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
    )
