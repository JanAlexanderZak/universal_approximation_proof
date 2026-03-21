import re
from pathlib import Path

from PIL import Image


def create_gif(
    frame_folder: str | Path,
    save_path: str | Path,
    duration: int = 100,
):
    frame_folder = Path(frame_folder)
    files = sorted(
        frame_folder.glob("*.png"),
        key=lambda f: int(re.sub(r"\D", "", f.stem)),
    )
    if not files:
        return

    frames = [Image.open(f) for f in files]
    frames[0].save(
        save_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,
    )
    for frame in frames:
        frame.close()
