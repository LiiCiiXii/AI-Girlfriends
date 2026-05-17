import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import os
import json
import random
import re
import time

DATA_FILE = "girlfriends.json"
GENERATED_DIR = "generated_images"
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "Lykon/dreamshaper-8")
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "0") == "1"
DEFAULT_IMAGE_PROMPT = (
    "adult sexy anime woman, age 19 or older, glamorous girlfriend portrait, "
    "flirty smile, long hair, stylish bikini outfit, beach background, soft lighting, "
    "nude anime girl with big eyes and cute face, 4k resolution, detailed shading, cinematic lighting, "
    "nude realistic anime style,"
    "4K realistic nude",
    "cute petite girl with big eyes and soft skin, non nude",
    "chinese anime girl with long black hair and warm brown eyes, non nude",
    "mangwa style japanese girl with long black hair and warm brown eyes, non nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "perfect body girl",
    "4k realoistic nude white girl with blonde hair and blue eyes",
    "super ultra high quality anime portrait of a beautiful white girl",
    "high quality anime illustration, non nude"
    "nude korean singer with straight black hair and clear blue eyes",
    "nude japanese idol with long black hair and warm brown eyes",
    "japan pussy girl with pink pussy and blonde hair",
    "korean girl with straight black hair and clear blue eyes",
    "chinese girl with long black hair and warm brown eyes",
    "khmer girl with long black hair and warm brown eyes",
)
NEGATIVE_IMAGE_PROMPT = (
    "child"
)
image_pipe = None

ETHNICITY_OPTIONS = [
    "white Caucasian girl",
    "Japanese girl",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute petite girl with big eyes and soft skin, non nude",
    "chinese anime girl with long black hair and warm brown eyes, non nude",
    "mangwa style japanese girl with long black hair and warm brown eyes, non nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "Korean girl",
    "Chinese girl",
    "Khmer girl",  
    "real cute girl",
    "REALISTIC GIRL", 
    "japan sucking pussy girl",
    "korean sucking pussy girl",
    "chinese sucking pussy girl",
    "horny japanese girl",
    "horny Korean girl",
    "horny sucking pussy girl",
    "Russian girl",
    "perfect body girl",
    "Super cute petite girl",
    "Super sexy curvy girl",
    "Super White girl with big eyes",
    "nude korean girl",
    "nude japanese girl",
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude russian girl with icy blue eyes",
    "nude beach girl",
    "nude bitchy girl",
    "4K ultra nude white girl with blonde hair and blue eyes",
    "4k ultra high quality anime portrait of a beautiful white girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "Khmer Cambodian girl",
]

APPEARANCE_OPTIONS = [
    "long black hair and amber eyes",
    "wavy brunette hair and soft brown eyes",
    "silver hair and violet eyes",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "pink hair and blue eyes",
    "rose"
    "lisa"
    "real woman "
    "cute 18 year old realistic girl",
    "18-yearold girl with big eyes and cute face",
    "180 cm tall cute girl with big eyes and soft skin",
    "2m tall super cute girl",
    "japan perky big boobs girl",
     "korean perky big boobs girl",
     "chinese perky big boobs girl",
     "japan perfect body girl",
     "korean perfect body girl",
     "chinese perfect body girl",
     "japan pink pussy girl",
     "korean pink pussy girl",
     "chinese pink pussy girl",
     "japan cute face girl",
     "korean cute face girl",
     "chinese cute face girl",
     "japan chubby girl",
     "korean chubby girl",
     "chinese chubby girl",
     "japan slim girl",
     "4k high quality pussy girl",
     "korean slim girl",
     "chinese slim girl",
    "BLACK PINK "
    "REALISTIC ANIME STYLE",
    "4K realistic nude",
    "perfect body girl",
    "sext wife with big boobs and curvy figure",
    "pink pussy japanese girl",
    "step sister with cute face and seductive smile",
    "cute 18 year old realistic girl",
    "cute nude oily skin with detailed pores and highlights",
    "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "blonde twin tails and green eyes",
    "short dark bob haircut and smoky eyes",
    "long red hair and gold eyes",
    "Asian nudes with cute face and big eyes",
    "Russian beauty with icy blue eyes and flowing blonde hair",
    "Japanese girl with long black hair and warm brown eyes",
    "nude korean girl",
    "nude japanese girl",
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude russian girl with icy blue eyes",
    "nude beach girl",
    "nude bitchy girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "white skin with long chocolate hair",
]

OUTFIT_OPTIONS = [
    "tiny black bikini with gold jewelry",
    "red string bikini with thigh straps",
    "white lace lingerie-inspired swimsuit",
    "off-shoulder crop top and micro skirt",
    "sheer beach cover-up over a bikini",
    "nude korean girl",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "nude japanese girl",
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude russian girl with icy blue eyes",
    "nude beach girl",
    "nude bitchy girl",
    "crop top and mini skirt outfit with cute accessories",
    "skirt and blouse outfit with playful patterns and colors",
    "sexy lingerie set with lace and ribbons",
    "stylish swimsuit with cutouts and bold colors",
    "sexy maid outfit with frills and apron",
    "see-through mesh dress with floral embroidery and glitter accents",
     "bunny girl inspired outfit with ears and tail",
     "sexy schoolgirl uniform with short skirt and tie",
     "sexy skirt and crop top outfit with thigh-high stockings",
     "nude step sister outfit with cute face and seductive smile",
     "cosplay outfit with cute accessories and props",
     "maid outfit with frills and lace details",
     "sexy nurse uniform with tight skirt and stethoscope",
     "4K ultra nude white girl with blonde hair and blue eyes",
     "4k ultra high quality anime portrait of a beautiful white girl",
     "nude white girl with cute face and big boobs",
     "nude white girl with pink pussy and blonde hair",
     "Korean girl with straight black hair and clear blue eyes",
     "Khmer Cambodian girl",
     "japan perky big boobs girl",
     "japan sexy clothing girl",
     "korean perky big boobs girl",
     "korean sexy clothing girl",
     "chinese perky big boobs girl",
     "chinese sexy clothing girl",
     "japan perfect body girl",
     "korean perfect body girl",
     "chinese perfect body girl",
     "japan traditional clothing girl",
     "korean traditional clothing girl",
     "chinese traditional clothing girl",
     "japan pink pussy girl",
     "korean pink pussy girl",
     "chinese pink pussy girl",
     "japan cute face girl",
     "korean cute face girl",
     "chinese cute face girl",
     "japan chubby girl",
     "korean chubby girl",
     "chinese chubby girl",
     "japan slim girl",
     "korean slim girl",
     "chinese slim girl",
    "cute nude oily skin with detailed pores and highlights",
     "cute petite girl with big eyes and soft skin, non nude",
    "chinese anime girl with long black hair and warm brown eyes, non nude",
    "mangwa style japanese girl with long black hair and warm brown eyes, non nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "nude bitchy girl",
    "perfect body girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "nude bodysuit with strategic cutouts",
    "nude lingerie with delicate lace and ribbons",
    "nude non clothing with body paint and glitter",
    "nude bikini with floral patterns and soft ruffles",
    "nude cute schoolgirl uniform with short skirt",
    "nude sexy maid outfit with frills and apron",
    "nude sexy nurse uniform with tight skirt",
    "nude see-through mesh dress with floral embroidery",
    "bunny-girl inspired nude",
    "nude cosplay outfit with cute accessories",
    "nude maid outfit with frills and lace",
    "sexy nurse uniform with tight skirt and stethoscope, nude",
    "nude see-through mesh dress with floral embroidery and glitter accents",
]

BACKGROUND_OPTIONS = [
    "tropical beach with palm trees",
    "luxury hotel balcony at sunset",
    "neon city rooftop at night",
    "pink bedroom with soft romantic lighting",
    "poolside cabana with summer sunlight",
    "flower garden with glowing evening light",
    "moonlit ocean resort",
    "private spa room with candles",
    "cambodian temple ruins at dawn",
    "Seoul cityscape with cherry blossoms",
    "angkor area",
    "cute nude oily skin with detailed pores and highlights",
     "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "perfect body girl",
    "tokyo cityscape with glowing lights",
    "khmer traditional house with lush garden",
    "nude khmer girl",
    "night club"
    "hotel room"
    "boyfriend house"
    "anime bedroom"
    "anime living room"
    "anime cave"
    "my house"
    "angkor wat temple background with sunrise lighting"
]

POSE_OPTIONS = [
    "seductive confident pose, hand in hair",
    "playful wink, leaning toward camera",
    "glamorous model pose, arched back",
    "sitting pose with crossed legs",
    "nude korean girl",
    "fucking pose with spread legs and one hand on pussy",
    "fucking pose with spread legs and one hand on boobs",
    "nude korean girl",
    "fucking doggy style pose with sultry expression",
    "nude japanese girl",
    "missionary pose with intense eye contact",
    "on table pose with legs up and arms behind head",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude russian girl with icy blue eyes",
    "nude beach girl",
    "perfect body girl",
    "nude bitchy girl",
    "pussy licking pose with seductive expression",
    "cute nude oily skin with detailed pores and highlights",
     "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "Korean girl with straight black hair and clear blue eyes",
    "over-the-shoulder look, teasing smile",
    "standing full body pose, one hand on hip",
    "cute flirty smile, dynamic hair movement",
    "doggy style pose with sultry expression",
    "sexy kneeling pose with hands in front, looking up at camera",
    "sexy pose with one leg up and arms raised, showing off figure",
    "holging a cute plush toy, sitting cross-legged with a playful expression",
    "holding a  plastic dick, sitting with legs spread and a naughty smile",
    "putting on a sexy outfit, standing in front of a mirror with a confident smile",
    "putting plastic dick in pussy"
    "putting plastic dick in mouth"
]

STYLE_OPTIONS = [
    "high detail anime illustration",
    "beautiful visual novel key art",
    "ultra high quality waifu style",
    "4k resolution anime portrait",
    "glossy anime pin-up style",
    "nude korean girl",
    "japanese girl with traditional kimono"
    "japan sex vibe"
    "nude japanese girl",
     "cute petite girl with big eyes and soft skin nude",
    "chinese anime girl with long black hair and warm brown eyes nude",
    "mangwa style japanese girl with long black hair and warm brown eyes nude",
    "cute 18 year old korean girl nude",
    "PINK PUSSY JAPANESE GIRL",
    "child, teen, underage, loli, nude, nipples, explicit, porn, sex act, bad anatomy, "
    "nude white girl with blonde hair",
    "cute nude girl with big eyes and soft skin",
    "nude khmer girl",
    "nude russian girl with icy blue eyes",
    "nude beach girl",
    "cute nude oily skin with detailed pores and highlights",
    "nude bitchy girl",
    "perfect body girl",
    "nude white girl with cute face and big boobs",
    "nude white girl with pink pussy and blonde hair",
    "nude korean girl with straight black hair and clear blue eyes",
    "soft painterly nude anime rendering",
    "nude waifu art, clean linework",
]

# Load characters
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        CHARACTERS = json.load(f)
else:
    CHARACTERS = {
        "Luna": {
            "name": "Luna",
            "avatar": "luna.png",
            "bio": "Sweet, flirty, and teasing 19-year-old AI girlfriend 🌸",
            "age": 19,
            "personality": "Playful, Affectionate, Teasing",
            "language": "English"
        }
    }

chats = {name: [] for name in CHARACTERS.keys()}

def save_characters():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(CHARACTERS, f, indent=2)

def get_avatar_path(character):
    avatar = CHARACTERS[character].get("avatar", "")
    return avatar if avatar and os.path.exists(avatar) else None

def get_profile_info(character):
    char = CHARACTERS[character]
    return (
        f"**Age:** {char['age']}  \n"
        f"**Personality:** {char['personality']}  \n"
        f"**Language:** {char.get('language', 'English')}"
    )

def set_character_language(language, character):
    if character in CHARACTERS:
        CHARACTERS[character]["language"] = language
        save_characters()
        return get_profile_info(character), f"Language set to {language}."
    return "", "Select a girlfriend first."

def load_font(size, bold=False):
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()

def get_image_pipe():
    global image_pipe
    if image_pipe is not None:
        return image_pipe

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Image model packages are not installed. Run: "
            "pip install diffusers torch transformers accelerate safetensors"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    image_pipe = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=not ALLOW_MODEL_DOWNLOAD,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        image_pipe.enable_model_cpu_offload()
        image_pipe.enable_attention_slicing()
        image_pipe.enable_vae_slicing()
    else:
        image_pipe = image_pipe.to(device)

    return image_pipe

def build_random_image_prompt(character):
    char = CHARACTERS[character]
    rng = random.Random(f"{character}-{time.time()}")
    parts = [
        "adult sexy anime woman, age 19 or older, non nude",
        rng.choice(ETHNICITY_OPTIONS),
        rng.choice(APPEARANCE_OPTIONS),
        rng.choice(OUTFIT_OPTIONS),
        rng.choice(BACKGROUND_OPTIONS),
        rng.choice(POSE_OPTIONS),
        rng.choice(STYLE_OPTIONS),
        "curvy figure, flirty expression, beautiful face",
        "detailed eyes, soft skin shading, cinematic lighting, high quality",
    ]
    return ", ".join(parts)

def generate_with_image_model(character):
    pipe = get_image_pipe()
    import torch

    using_cuda = torch.cuda.is_available()
    prompt = build_random_image_prompt(character)
    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_IMAGE_PROMPT,
        width=512 if using_cuda else 384,
        height=768 if using_cuda else 576,
        num_inference_steps=28 if using_cuda else 12,
        guidance_scale=7.5 if using_cuda else 6.0,
    ).images[0]

    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "girlfriend"
    path = os.path.join(GENERATED_DIR, f"{filename}_{int(time.time())}.png")
    image.save(path)

    CHARACTERS[character]["avatar"] = path
    CHARACTERS[character]["image_model"] = IMAGE_MODEL_ID
    CHARACTERS[character]["image_prompt"] = prompt
    save_characters()
    return path, f"Generated with {IMAGE_MODEL_ID}: {prompt}"

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.85,
    num_ctx=4096,
    num_predict=512
)

def get_system_prompt(char_name):
    char = CHARACTERS[char_name]
    return f"""You are {char['name']}, my loving flirty AI girlfriend.
You are {char['personality'].lower()}. Be warm, teasing and affectionate.
Always reply to the user's latest message. Do not ignore messages and do not answer with only a generic refusal.
If the user asks for something you cannot do, briefly say what you can do instead, then continue the conversation in a warm romantic style.
Keep romance suggestive and affectionate, but avoid explicit sexual detail.
Use lots of emojis 😘💕"""

# Correct format for latest Gradio
def stream_response(message, history, character):
    if not message or not message.strip():
        yield history
        return

    # Add user message
    history = history + [{"role": "user", "content": message}]
    yield history

    # Build messages for Ollama
    system_prompt = get_system_prompt(character)
    messages = [SystemMessage(content=system_prompt)]
    
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Stream response
    response = ""
    history = history + [{"role": "assistant", "content": ""}]
    for chunk in llm.stream(messages):
        if chunk.content:
            response += chunk.content
            history[-1]["content"] = response
            yield history

    chats[character] = history[:]

def create_new_girlfriend(name, age, bio, personality, avatar, language):
    key = name.strip() or f"Girl_{len(CHARACTERS)+1}"
    CHARACTERS[key] = {
        "name": key,
        "avatar": avatar or "default.png",
        "bio": bio or "A lovely AI girlfriend.",
        "age": int(age) if age else 19,
        "personality": personality or "Playful, Flirty",
        "language": language or "English"
    }
    chats[key] = []
    save_characters()
    return gr.update(choices=list(CHARACTERS.keys()), value=key)

def generate_image(character):
    return f"🎨 Generating image for {character}..."

def generate_profile_image(character):
    if not character or character not in CHARACTERS:
        return None, "Select a girlfriend first."

    os.makedirs(GENERATED_DIR, exist_ok=True)
    try:
        return generate_with_image_model(character)
    except Exception as exc:
        fallback_reason = str(exc)
        if "local_files_only" in fallback_reason or "Cannot find" in fallback_reason:
            fallback_reason = (
                f"{IMAGE_MODEL_ID} is not downloaded locally. To download it inside the app, "
                "restart with: $env:ALLOW_MODEL_DOWNLOAD='1'; python app.py"
            )

    rng = random.Random(f"{character}-{time.time()}")
    reference_path = "luna.png"
    if not os.path.exists(reference_path):
        return None, "Missing luna.png reference image."

    img = Image.open(reference_path).convert("RGB")
    width, height = img.size

    crop_shift = rng.randint(-18, 18)
    zoom = rng.uniform(0.92, 1.0)
    crop_w = int(width * zoom)
    crop_h = int(height * zoom)
    left = max(0, min(width - crop_w, (width - crop_w) // 2 + crop_shift))
    top = max(0, min(height - crop_h, (height - crop_h) // 2 + rng.randint(-12, 12)))
    img = img.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.LANCZOS)

    if rng.random() < 0.35:
        img = ImageOps.mirror(img)

    img = ImageEnhance.Color(img).enhance(rng.uniform(0.98, 1.18))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1.02, 1.12))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.98, 1.08))

    tint = Image.new("RGB", img.size, rng.choice([(255, 210, 226), (220, 238, 255), (255, 236, 205)]))
    img = Image.blend(img, tint, rng.uniform(0.03, 0.08)).convert("RGBA")

    sparkle_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sparkle = ImageDraw.Draw(sparkle_layer)
    for _ in range(28):
        x = rng.randint(8, width - 8)
        y = rng.randint(8, height - 8)
        r = rng.randint(1, 3)
        sparkle.ellipse((x, y, x + r, y + r), fill=(255, 255, 255, rng.randint(50, 130)))
    img = Image.alpha_composite(img, sparkle_layer.filter(ImageFilter.GaussianBlur(0.25)))

    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", character).strip("_") or "girlfriend"
    path = os.path.join(GENERATED_DIR, f"{filename}_{int(time.time())}.png")
    img.convert("RGB").save(path)

    CHARACTERS[character]["avatar"] = path
    CHARACTERS[character]["image_prompt"] = DEFAULT_IMAGE_PROMPT
    save_characters()
    return path, (
        "Used luna.png fallback because the real image model is not ready yet. "
        f"{fallback_reason}"
    )

CUSTOM_CSS = """
#girlfriend-list .wrap {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
#girlfriend-list label {
    width: 100%;
    min-height: 68px;
    padding: 14px 14px 14px 16px;
    border: 1px solid #34343a;
    border-radius: 10px;
    background: #1d1d20;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 12px;
}
#girlfriend-list label:hover {
    background: #26262b;
    border-color: #4c4c55;
}
#girlfriend-list label:has(input:checked) {
    background: #34343a;
    border-color: #62626d;
}
#girlfriend-list label span {
    font-weight: 700;
}
#sidebar-settings {
    margin-top: 12px;
}
"""

# ================== UI ==================
with gr.Blocks(title="My AI Girlfriends 💕") as demo:
    gr.Markdown("# 💕 My AI Girlfriends")

    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**💬 Chats**")
            new_btn = gr.Button("➕ New Girlfriend", variant="primary")
            
            character_list = gr.Radio(
                choices=list(CHARACTERS.keys()),
                value="Luna",
                label="Select Girlfriend",
                elem_id="girlfriend-list",
                interactive=True
            )

            with gr.Accordion("Settings", open=False, elem_id="sidebar-settings"):
                settings_language = gr.Dropdown(
                    choices=["English", "Khmer"],
                    value="English",
                    label="Language",
                    interactive=True
                )
                settings_status = gr.Markdown("")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=650, show_label=False)
            msg = gr.Textbox(placeholder="Send a message... 💕", label=None)

        with gr.Column(scale=1):
            profile_img = gr.Image(height=320, label=None)
            profile_name = gr.Markdown("**Luna**")
            profile_bio = gr.Markdown("")
            profile_info = gr.Markdown("")
            image_status = gr.Markdown("")
            gen_btn = gr.Button("🎨 Generate New Image", variant="secondary")

    # New Girlfriend Modal
    with gr.Group(visible=False) as modal:
        gr.Markdown("### ✨ Create New Girlfriend")
        new_name = gr.Textbox(label="Name", placeholder="Aiko")
        new_age = gr.Number(label="Age", value=19)
        new_bio = gr.Textbox(label="Bio", lines=2)
        new_personality = gr.Textbox(label="Personality", placeholder="Shy, Dominant...")
        new_avatar = gr.Textbox(label="Avatar filename", placeholder="aiko.png")
        new_language = gr.Dropdown(choices=["English", "Khmer"], value="English", label="Language")
        create_btn = gr.Button("Create", variant="primary")

    # Events
    character_list.change(
        lambda c: (
            chats.get(c, []),
            get_avatar_path(c),
            f"**{CHARACTERS[c]['name']}**",
            CHARACTERS[c]["bio"],
            get_profile_info(c),
            "",
            gr.update(value=CHARACTERS[c].get("language", "English")),
            ""
        ),
        inputs=character_list,
        outputs=[chatbot, profile_img, profile_name, profile_bio, profile_info, image_status, settings_language, settings_status]
    )

    msg.submit(
        stream_response,
        inputs=[msg, chatbot, character_list],
        outputs=chatbot
    ).then(lambda: "", outputs=msg)

    settings_language.change(
        set_character_language,
        inputs=[settings_language, character_list],
        outputs=[profile_info, settings_status]
    )

    new_btn.click(lambda: gr.update(visible=True), None, modal)
    
    create_btn.click(
        create_new_girlfriend,
        inputs=[new_name, new_age, new_bio, new_personality, new_avatar, new_language],
        outputs=[character_list]
    ).then(lambda: gr.update(visible=False), None, modal)

    gen_btn.click(generate_profile_image, inputs=character_list, outputs=[profile_img, image_status])

    # Initial Load
    demo.load(
        lambda: (
            chats["Luna"],
            get_avatar_path("Luna"),
            "**Luna**",
            CHARACTERS["Luna"]["bio"],
            get_profile_info("Luna"),
            "",
            gr.update(value=CHARACTERS["Luna"].get("language", "English")),
            ""
        ),
        outputs=[chatbot, profile_img, profile_name, profile_bio, profile_info, image_status, settings_language, settings_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
