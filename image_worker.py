import json
import os
import random
import re
import sys
import time

DATA_FILE = "girlfriends.json"
GENERATED_DIR = "generated_images"
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "segmind/tiny-sd")
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "1") == "1"
IMAGE_USE_SAFETENSORS = os.getenv("IMAGE_USE_SAFETENSORS", "0") == "1"
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "384"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "576"))
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "18"))
IMAGE_GUIDANCE = float(os.getenv("IMAGE_GUIDANCE", "7.5"))

NEGATIVE_IMAGE_PROMPT = (
    "child, teen, underage, loli, nude, nipples, topless, bare breasts, explicit, porn, sex act, "
    "realistic, photorealistic, photo, semi-realistic, 3d render, doll face, old woman, masculine, bad anatomy, "
    "bad hands, extra fingers, missing fingers, fused fingers, deformed face, ugly, "
    "low quality, blurry, watermark, text, logo, cropped head, out of frame"
)

ACTIVITY_SCENES = {
    "selfie": "taking a cute mirror selfie, flirty smile, cozy bedroom",
    "photo": "posing for a glamorous girlfriend photo, flirty smile, soft lighting",
    "picture": "posing for a glamorous girlfriend photo, flirty smile, soft lighting",
    "pic": "posing for a glamorous girlfriend photo, flirty smile, soft lighting",
    "eating": "eating dessert at a cute cafe, playful smile, leaning toward camera",
    "eat": "eating dessert at a cute cafe, playful smile, leaning toward camera",
    "cooking": "cooking dinner in a bright kitchen, holding a mixing bowl, playful smile",
    "cook": "cooking dinner in a bright kitchen, holding a mixing bowl, playful smile",
    "drinking": "drinking iced coffee at a cafe, flirty smile, looking at camera",
    "sleeping": "relaxing on a sofa with a blanket, sleepy smile, cozy pose",
    "what are you doing": "taking a casual selfie while relaxing at home, flirty smile",
    "what r u doing": "taking a casual selfie while relaxing at home, flirty smile",
    "wyd": "taking a casual selfie while relaxing at home, flirty smile",
}

ETHNICITY_OPTIONS = [
    "Khmer Cambodian woman",
    "Japanese woman",
    "Korean woman",
    "Chinese woman",
    "white Caucasian woman",
    "Russian woman",
]

APPEARANCE_OPTIONS = [
    "long black hair and amber eyes",
    "wavy brunette hair and soft brown eyes",
    "silver hair and violet eyes",
    "pink hair and blue eyes",
    "blonde twin tails and green eyes",
    "short dark bob haircut and smoky eyes",
    "long red hair and gold eyes",
    "tan skin with long chocolate hair",
]

OUTFIT_OPTIONS = [
    "tiny black bikini with gold jewelry, non nude",
    "red string bikini with thigh straps, non nude",
    "white lace swimsuit, non nude",
    "off-shoulder crop top and micro skirt",
    "sheer beach cover-up over a bikini, non nude",
    "glossy fitted bodysuit, non nude",
    "low-cut cocktail dress with high slit",
    "bunny-girl inspired leotard, non nude",
]

BACKGROUND_OPTIONS = [
    "bright kitchen",
    "cozy bedroom with soft romantic lighting",
    "sunny cafe",
    "tropical beach with palm trees",
    "neon city rooftop at night",
    "flower garden with glowing evening light",
    "angkor wat temple background with sunrise lighting",
]

POSE_OPTIONS = [
    "seductive confident pose, hand in hair",
    "playful wink, leaning toward camera",
    "glamorous model pose, arched back",
    "sitting pose with crossed legs",
    "over-the-shoulder look, teasing smile",
    "standing full body pose, one hand on hip",
    "cute flirty smile, dynamic hair movement",
    "sexy kneeling pose with hands in front, looking up at camera",
    "sexy pose with one leg up and arms raised, showing off figure",
]

STYLE_OPTIONS = [
    "high detail anime illustration",
    "beautiful visual novel key art",
    "ultra high quality waifu style",
    "4k anime portrait",
    "glossy anime pin-up style",
    "soft painterly anime rendering",
    "vibrant waifu art, clean linework",
]


def emit(payload):
    print("IMAGE_RESULT:" + json.dumps(payload, ensure_ascii=False), flush=True)


def load_characters():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_activity_scene(message):
    text = (message or "").lower()
    for trigger, scene in ACTIVITY_SCENES.items():
        if trigger in text:
            return scene
    cleaned = re.sub(r"[^a-zA-Z0-9 ,_-]+", " ", message or "").strip()
    if cleaned:
        return cleaned[:160]
    return "taking a cute casual selfie, cozy room, warm smile"


def get_scene_details(message):
    text = (message or "").lower()
    if not text.strip():
        return (
            "anime beach selfie portrait, upper body crop",
            "cute red bikini top, covered chest",
            "tropical beach, palm trees",
        )
    if "cook" in text:
        return (
            "cooking in bright kitchen, holding mixing bowl",
            "cute apron, off-shoulder crop top, short skirt",
            "modern kitchen counter",
        )
    if "eat" in text:
        return (
            "eating dessert at cute cafe",
            "low-cut summer dress",
            "sunny cafe table",
        )
    if "drink" in text:
        return (
            "drinking iced coffee at cafe",
            "off-shoulder crop top, micro skirt",
            "sunny cafe table",
        )
    if "sleep" in text:
        return (
            "relaxing on sofa with blanket",
            "soft fitted loungewear",
            "cozy bedroom, romantic lighting",
        )
    return get_activity_scene(message), None, None


def build_default_prompt(character, rng, age):
    return ", ".join(
        [
            f"adult sexy anime woman, age {age} or older, non nude",
            rng.choice(ETHNICITY_OPTIONS),
            rng.choice(APPEARANCE_OPTIONS),
            rng.choice(OUTFIT_OPTIONS),
            rng.choice(BACKGROUND_OPTIONS),
            rng.choice(POSE_OPTIONS),
            rng.choice(STYLE_OPTIONS),
            "curvy figure, flirty expression, beautiful face",
            "detailed eyes, soft skin shading, cinematic lighting, high quality",
        ]
    )


def build_command_prompt(character, rng, age, request_text):
    activity, required_outfit, required_background = get_scene_details(request_text)
    return ", ".join(
        [
            f"adult sexy anime woman, age {age} or older, non nude",
            activity,
            rng.choice(ETHNICITY_OPTIONS),
            rng.choice(APPEARANCE_OPTIONS),
            required_outfit or rng.choice(OUTFIT_OPTIONS),
            required_background or rng.choice(BACKGROUND_OPTIONS),
            rng.choice(POSE_OPTIONS),
            rng.choice(STYLE_OPTIONS),
            "curvy figure, flirty expression, beautiful face",
            "detailed eyes, soft skin shading, cinematic lighting, high quality",
        ]
    )


def build_prompt(character, request_text):
    characters = load_characters()
    char = characters.get(character, {})
    rng = random.Random(f"{character}-{request_text}-{time.time()}")
    age = max(int(char.get("age", 19) or 19), 19)
    if not (request_text or "").strip():
        return build_default_prompt(character, rng, age)
    return build_command_prompt(character, rng, age, request_text)


def generate(character, request_text):
    import torch
    from diffusers import StableDiffusionPipeline

    os.makedirs(GENERATED_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=IMAGE_USE_SAFETENSORS,
        local_files_only=not ALLOW_MODEL_DOWNLOAD,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    if device == "cuda":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)

    prompt = build_prompt(character, request_text)
    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_IMAGE_PROMPT,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        num_inference_steps=IMAGE_STEPS,
        guidance_scale=IMAGE_GUIDANCE,
    ).images[0]

    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "girlfriend"
    path = os.path.join(GENERATED_DIR, f"{filename}_{int(time.time())}.png")
    image.save(path)
    return path, f"Generated with {IMAGE_MODEL_ID}: {prompt}"


if __name__ == "__main__":
    character_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    request_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        image_path, status = generate(character_arg, request_arg)
        emit({"ok": True, "path": image_path, "status": status})
    except BaseException as exc:
        emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise
