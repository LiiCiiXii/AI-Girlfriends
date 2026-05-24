import os
import random
import re
import shutil
import subprocess
import tempfile
import time

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


GENERATED_VIDEO_DIR = "generated_videos"
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 1024
DEFAULT_DURATION = 4.0
DEFAULT_FPS = 24

VIDEO_PRESETS = {
    "Dance": "upbeat dancing motion, rhythmic body bounce, side to side sway, playful camera energy, party lights",
    "Walk": "walking forward motion, smooth step rhythm, gentle side sway, street-style camera tracking",
    "Run": "running motion, fast energetic shake, strong forward movement, dynamic action blur",
    "Sleep": "sleeping peacefully, slow breathing motion, calm dreamy lighting, soft eyelid mood",
    "Drink": "drinking from a cup, tiny lean forward, hand-to-mouth gesture illusion, cozy cafe light",
    "Wave": "waving hello, cheerful side sway, hand wave illusion, friendly smile energy",
    "Hair wind": "wind blowing hair, soft drifting motion, cinematic breeze, elegant portrait",
    "Laugh": "laughing happily, bright bounce, playful head tilt, warm glow",
    "Heartbeat": "romantic heartbeat pulse, soft glow, tiny zoom pulses, dreamy mood",
    "Cinematic push in": "slow cinematic push-in, gentle face-focused camera movement, soft romantic lighting, subtle hair movement",
    "Soft portrait motion": "soft portrait animation, tiny breathing zoom, dreamy glow, calm elegant motion",
    "Slow left pan": "slow camera pan to the left, light parallax feeling, smooth portrait video",
    "Slow right pan": "slow camera pan to the right, light parallax feeling, smooth portrait video",
    "Dream glow": "dreamy glowing atmosphere, slow zoom, soft bloom, romantic mood",
    "Custom only": "",
}

ACTION_HINTS = {
    "Dance": "Rhythmic bounce, side sway, camera tilt, colored party light.",
    "Walk": "Step-by-step sway with a slow forward tracking feel.",
    "Run": "Fast shake, stronger forward movement, action blur.",
    "Sleep": "Very slow breathing pulse, dim dreamy vignette.",
    "Drink": "Leans forward and adds a small cup prop motion.",
    "Wave": "Adds a simple waving hand/arc overlay with cheerful bounce.",
    "Hair wind": "Soft breeze streaks and drifting portrait motion.",
    "Laugh": "Bright playful bounce and head tilt feeling.",
    "Heartbeat": "Romantic pulse zoom and glow beats.",
}

CUSTOM_CSS = """
:root {
    --panel-border: rgba(255, 255, 255, 0.12);
}
.gradio-container {
    max-width: 1440px !important;
}
#hero {
    padding: 26px;
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(255, 99, 132, 0.14), rgba(66, 211, 183, 0.10));
    margin-bottom: 14px;
}
#hero h1 {
    margin-bottom: 6px;
    font-size: 42px;
    line-height: 1.05;
}
#hero p {
    margin: 0;
    color: #b8bac7;
    font-size: 16px;
}
#status-box {
    min-height: 52px;
    padding: 12px 14px;
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.05);
}
#generate-btn {
    min-height: 54px;
    font-weight: 800;
}
"""


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "image_video"


def get_uploaded_path(uploaded_file):
    if not uploaded_file:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("path") or uploaded_file.get("name")
    return getattr(uploaded_file, "name", None) or getattr(uploaded_file, "path", None)


def load_preview(uploaded_file):
    path = get_uploaded_path(uploaded_file)
    if not path or not os.path.exists(path):
        return None, None, "Choose an image file first."

    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        return None, None, f"Could not open that image: {exc}"

    return path, image, f"Loaded image: {os.path.basename(path)}"


def preset_to_prompt(preset):
    return VIDEO_PRESETS.get(preset, "")


def fit_cover(image, width, height):
    image = ImageOps.exif_transpose(image).convert("RGB")
    source_w, source_h = image.size
    scale = max(width / source_w, height / source_h)
    resized = image.resize((int(source_w * scale), int(source_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def wave_value(progress, cycles=1.0, phase=0.0):
    return np.sin((progress * cycles + phase) * np.pi * 2)


def pulse_value(progress, cycles=1.0, phase=0.0):
    return (wave_value(progress, cycles, phase) + 1.0) / 2.0


def transform_frame(base_image, zoom, x_offset, y_offset, angle):
    width, height = base_image.size
    zoom = max(1.0, zoom)
    crop_w = max(1, int(width / zoom))
    crop_h = max(1, int(height / zoom))
    left = max(0, min(width - crop_w, (width - crop_w) // 2 + int(x_offset)))
    top = max(0, min(height - crop_h, (height - crop_h) // 2 + int(y_offset)))
    frame = base_image.crop((left, top, left + crop_w, top + crop_h))
    frame = frame.resize((width, height), Image.Resampling.LANCZOS)
    if abs(angle) > 0.01:
        frame = frame.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    return frame


def add_vignette(frame, alpha=70):
    width, height = frame.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((-width * 0.16, -height * 0.08, width * 1.16, height * 1.06), fill=255)
    mask = ImageOps.invert(mask.filter(ImageFilter.GaussianBlur(width // 8)))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, int(alpha)))
    frame_rgba = frame.convert("RGBA")
    frame_rgba.alpha_composite(Image.composite(overlay, Image.new("RGBA", (width, height), (0, 0, 0, 0)), mask))
    return frame_rgba.convert("RGB")


def add_light_sweep(frame, progress, color=(255, 110, 150), alpha=44):
    width, height = frame.size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x = int((progress * 1.4 - 0.2) * width)
    draw.polygon(
        [(x - 180, 0), (x + 20, 0), (x + 220, height), (x + 20, height)],
        fill=(*color, alpha),
    )
    return Image.alpha_composite(frame.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(18))).convert("RGB")


def add_speed_lines(frame, progress, strength):
    width, height = frame.size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line_count = int(12 + 30 * strength)
    for i in range(line_count):
        y = int(((i / line_count) + progress * 1.8) % 1.0 * height)
        x1 = int(width * (0.05 + 0.20 * ((i * 7) % 5) / 5))
        draw.line((x1, y, width + 80, y - int(60 + 90 * strength)), fill=(255, 255, 255, 42), width=2)
    return Image.alpha_composite(frame.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(1.2))).convert("RGB")


def add_wind_streaks(frame, progress, strength):
    width, height = frame.size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i in range(18):
        y = int(((i * 0.071 + progress * 0.7) % 1) * height)
        x = int(((i * 0.193 + progress * 0.35) % 1) * width)
        draw.arc((x - 120, y - 28, x + 120, y + 42), 200, 340, fill=(210, 245, 255, int(35 + 35 * strength)), width=2)
    return Image.alpha_composite(frame.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(0.8))).convert("RGB")


def add_drink_prop(frame, progress, strength):
    width, height = frame.size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    lift = pulse_value(progress, 1.0)
    x = int(width * (0.74 - 0.08 * lift))
    y = int(height * (0.72 - 0.25 * lift))
    cup_w = int(width * 0.075)
    cup_h = int(height * 0.075)
    draw.rounded_rectangle((x, y, x + cup_w, y + cup_h), radius=8, fill=(245, 245, 250, 210), outline=(255, 180, 205, 230), width=3)
    draw.ellipse((x + 5, y - 6, x + cup_w - 5, y + 8), fill=(130, 76, 45, 180))
    return Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")


def add_wave_prop(frame, progress, strength):
    width, height = frame.size
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    swing = wave_value(progress, 3.0)
    cx = int(width * 0.76 + swing * width * 0.035 * strength)
    cy = int(height * 0.34 + wave_value(progress, 3.0, 0.2) * height * 0.02)
    radius = int(width * 0.045)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 218, 196, 185), outline=(255, 245, 235, 210), width=3)
    for arc in range(3):
        offset = arc * 22
        draw.arc((cx - 55 - offset, cy - 60 - offset, cx + 55 + offset, cy + 60 + offset), 290, 45, fill=(255, 255, 255, 70), width=2)
    return Image.alpha_composite(frame.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(0.25))).convert("RGB")


def action_motion(action, progress, prompt_text, strength):
    if action == "Dance":
        return {
            "zoom": 1.04 + 0.05 * pulse_value(progress, 4.0) * strength,
            "x": 42 * strength * wave_value(progress, 2.0),
            "y": 34 * strength * wave_value(progress, 4.0, 0.25),
            "angle": 4.0 * strength * wave_value(progress, 2.0, 0.1),
            "blur": 0.0,
        }
    if action == "Walk":
        return {
            "zoom": 1.03 + 0.10 * progress * strength,
            "x": 28 * strength * wave_value(progress, 2.0),
            "y": 18 * strength * abs(wave_value(progress, 2.0)),
            "angle": 1.8 * strength * wave_value(progress, 2.0, 0.1),
            "blur": 0.0,
        }
    if action == "Run":
        return {
            "zoom": 1.08 + 0.18 * progress * strength,
            "x": 52 * strength * wave_value(progress, 5.0),
            "y": 38 * strength * wave_value(progress, 10.0),
            "angle": 3.0 * strength * wave_value(progress, 5.0),
            "blur": 0.8 + 1.8 * strength,
        }
    if action == "Sleep":
        return {
            "zoom": 1.02 + 0.025 * pulse_value(progress, 0.8) * strength,
            "x": 4 * strength * wave_value(progress, 0.6),
            "y": 10 * strength * pulse_value(progress, 0.8),
            "angle": 0.6 * strength * wave_value(progress, 0.5),
            "blur": 0.2,
        }
    if action == "Drink":
        return {
            "zoom": 1.04 + 0.04 * pulse_value(progress, 1.0) * strength,
            "x": 14 * strength * wave_value(progress, 1.0),
            "y": -24 * strength * pulse_value(progress, 1.0),
            "angle": -1.8 * strength * pulse_value(progress, 1.0),
            "blur": 0.0,
        }
    if action == "Wave":
        return {
            "zoom": 1.035 + 0.03 * pulse_value(progress, 2.0) * strength,
            "x": 24 * strength * wave_value(progress, 1.8),
            "y": 16 * strength * wave_value(progress, 3.0),
            "angle": 2.2 * strength * wave_value(progress, 1.8),
            "blur": 0.0,
        }
    if action == "Hair wind":
        return {
            "zoom": 1.04 + 0.05 * progress * strength,
            "x": 30 * strength * wave_value(progress, 0.8),
            "y": 10 * strength * wave_value(progress, 1.4),
            "angle": 1.2 * strength * wave_value(progress, 0.6),
            "blur": 0.0,
        }
    if action == "Laugh":
        return {
            "zoom": 1.035 + 0.06 * pulse_value(progress, 3.0) * strength,
            "x": 18 * strength * wave_value(progress, 2.4),
            "y": 28 * strength * abs(wave_value(progress, 3.0)),
            "angle": 2.4 * strength * wave_value(progress, 2.4),
            "blur": 0.0,
        }
    if action == "Heartbeat":
        beat = max(pulse_value(progress, 2.0) ** 5, pulse_value(progress, 2.0, 0.18) ** 7)
        return {
            "zoom": 1.03 + 0.10 * beat * strength,
            "x": 4 * strength * wave_value(progress, 2.0),
            "y": 4 * strength * wave_value(progress, 2.0, 0.25),
            "angle": 0.4 * strength * wave_value(progress, 1.0),
            "blur": 0.0,
        }

    zoom_amount = 1.0 + (0.03 + 0.08 * strength) * progress
    if "push" in prompt_text or "zoom" in prompt_text or "dream" in prompt_text:
        zoom_amount = 1.0 + (0.06 + 0.12 * strength) * progress
    pan_strength = (12 + 48 * strength) * np.sin(progress * np.pi)
    if "left" in prompt_text:
        x_offset = -pan_strength
    elif "right" in prompt_text:
        x_offset = pan_strength
    else:
        x_offset = (6 + 20 * strength) * wave_value(progress, 1.0)
    return {
        "zoom": zoom_amount,
        "x": x_offset,
        "y": (4 + 16 * strength) * wave_value(progress, 1.0, 0.1),
        "angle": 0.0,
        "blur": 0.0,
    }


def make_frame(base_image, progress, prompt, strength, seed, action):
    rng = random.Random(seed)
    prompt_text = prompt.lower()
    motion = action_motion(action, progress, prompt_text, strength)
    frame = transform_frame(base_image, motion["zoom"], motion["x"], motion["y"], motion["angle"])

    if motion["blur"] > 0:
        frame = frame.filter(ImageFilter.GaussianBlur(motion["blur"]))

    if "glow" in prompt_text or "dream" in prompt_text or "soft" in prompt_text:
        glow = frame.filter(ImageFilter.GaussianBlur(radius=8 + 12 * strength))
        frame = Image.blend(frame, glow, 0.10 + 0.10 * strength)

    if action in {"Dance", "Laugh"}:
        frame = add_light_sweep(frame, (progress + rng.random() * 0.1) % 1, color=(255, 95, 150), alpha=int(35 + 55 * strength))
    if action == "Run":
        frame = add_speed_lines(frame, progress, strength)
    if action == "Sleep":
        frame = add_vignette(frame, alpha=int(85 + 50 * strength))
        frame = ImageEnhance.Brightness(frame).enhance(0.88)
    if action == "Drink":
        frame = add_drink_prop(frame, progress, strength)
    if action == "Wave":
        frame = add_wave_prop(frame, progress, strength)
    if action == "Hair wind":
        frame = add_wind_streaks(frame, progress, strength)
    if action == "Heartbeat":
        frame = add_light_sweep(frame, progress, color=(255, 60, 110), alpha=int(24 + 70 * strength * pulse_value(progress, 2.0)))

    brightness = 1.0 + 0.025 * np.sin(progress * np.pi * 2)
    color = 1.03 + 0.035 * np.sin(progress * np.pi)
    frame = ImageEnhance.Brightness(frame).enhance(brightness)
    frame = ImageEnhance.Color(frame).enhance(color)
    return frame


def generate_video(image_path, action, description, duration, fps, strength):
    if not image_path or not os.path.exists(image_path):
        return None, "Select an image first."

    prompt = (description or "").strip() or preset_to_prompt(action)
    if not prompt:
        prompt = VIDEO_PRESETS["Cinematic push in"]

    duration = max(1.0, float(duration or DEFAULT_DURATION))
    fps = max(8, min(60, int(fps or DEFAULT_FPS)))
    strength = max(0.0, min(1.0, float(strength or 0.45)))
    frame_count = max(1, int(duration * fps))

    os.makedirs(GENERATED_VIDEO_DIR, exist_ok=True)
    source = Image.open(image_path).convert("RGB")
    base = fit_cover(source, DEFAULT_WIDTH, DEFAULT_HEIGHT)

    seed = int(time.time())
    output_name = f"{safe_name(os.path.splitext(os.path.basename(image_path))[0])}_{seed}.mp4"
    output_path = os.path.abspath(os.path.join(GENERATED_VIDEO_DIR, output_name))

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None, "ffmpeg is required to create MP4 videos. Install ffmpeg or add it to PATH."

    with tempfile.TemporaryDirectory(prefix="image_to_video_") as frame_dir:
        for index in range(frame_count):
            progress = index / max(frame_count - 1, 1)
            frame = make_frame(base, progress, prompt, strength, seed, action)
            frame.save(os.path.join(frame_dir, f"frame_{index:05d}.png"))

        command = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frame_dir, "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return None, f"ffmpeg could not create the video:\n\n{result.stderr[-1200:]}"

    return output_path, (
        f"Video generated from uploaded image.\n\n"
        f"Action: {action}\n\n"
        f"Description: {prompt}\n\n"
        f"Saved: {output_path}"
    )


with gr.Blocks(title="Image to Video Studio") as demo:
    gr.HTML(
        """
        <section id="hero">
            <h1>Image to Video Studio</h1>
            <p>Select an image, choose an action like dance, walk, run, sleep, drink, or wave, then generate a short animated video from that uploaded image.</p>
        </section>
        """
    )

    selected_image_path = gr.State(value=None)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 1. Select Image")
            upload_btn = gr.UploadButton(
                "Select image from file",
                file_types=["image"],
                type="filepath",
                variant="primary",
            )
            image_preview = gr.Image(
                label="Uploaded image",
                height=520,
                type="pil",
                interactive=False,
            )

        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 2. Pick Animation")
            action = gr.Dropdown(
                choices=list(VIDEO_PRESETS.keys()),
                value="Dance",
                label="Animation action",
                interactive=True,
            )
            action_hint = gr.Markdown(ACTION_HINTS["Dance"])
            description = gr.Textbox(
                label="Description box",
                value=VIDEO_PRESETS["Dance"],
                lines=5,
                placeholder="Example: dance with playful bounce, colorful lights, side-to-side motion",
            )
            with gr.Row():
                duration = gr.Slider(1, 10, value=DEFAULT_DURATION, step=0.5, label="Duration")
                fps = gr.Slider(8, 60, value=DEFAULT_FPS, step=1, label="FPS")
            strength = gr.Slider(0, 1, value=0.45, step=0.05, label="Motion strength")
            generate_btn = gr.Button("Generate video from uploaded image", variant="primary", elem_id="generate-btn")
            status = gr.Markdown("Select an image to begin.", elem_id="status-box")

        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 3. Result")
            video_output = gr.Video(label="Generated video", height=520)

    upload_btn.upload(
        load_preview,
        inputs=upload_btn,
        outputs=[selected_image_path, image_preview, status],
    )

    action.change(
        lambda selected: (preset_to_prompt(selected), ACTION_HINTS.get(selected, "Custom camera and motion style.")),
        inputs=action,
        outputs=[description, action_hint],
    )

    generate_btn.click(
        generate_video,
        inputs=[selected_image_path, action, description, duration, fps, strength],
        outputs=[video_output, status],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CUSTOM_CSS)
