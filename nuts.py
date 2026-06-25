import os
import gc
import gradio as gr
import numpy as np
import spaces
import torch
import random
import base64
import json
import zipfile
import html as html_lib
import requests
from io import BytesIO
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
OUTPUT_DIR = "/content/qwen_batch_outputs"
LORA_DIR = "/content/qwen_loras"
PROMPT_SAVE_DIR = "/content/input"
MAX_OUTPUT_SIDE = 2048
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(PROMPT_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        torch_dtype=dtype,
        device_map="cuda",
    ),
    torch_dtype=dtype,
).to(device)

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

# One-time cleanup at startup only.
gc.collect()
torch.cuda.empty_cache()

ADAPTER_SPECS = {
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles",
    },
    "XY": {
        "repo": "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora",
        "weights": "qwen-image-edit-plus-nsfw-lora.safetensors",
        "adapter_name": "XY",
    },
    "CNFemale": {
        "url": "https://ai.x39.org/lora/cnfemale20.safetensors",
        "repo": LORA_DIR,
        "weights": "cnfemale20.safetensors",
        "adapter_name": "CNFemale",
    },
    "CNPussy": {
        "url": "https://ai.x39.org/lora/femalegenitals.safetensors",
        "repo": LORA_DIR,
        "weights": "femalegenitals.safetensors",
        "adapter_name": "CNPussy",
    },
    "banyetman": {
        "url": "https://ai.x39.org/banyetman/banyetman.safetensors",
        "repo": LORA_DIR,
        "weights": "banyetman.safetensors",
        "adapter_name": "banyetman",
    },
    "lock": {
        "url": "https://ai.x39.org/lora/banyetnutsv25k.safetensors",
        "repo": LORA_DIR,
        "weights": "banyetnutsv25k.safetensors",
        "adapter_name": "lock",
    },
    "beer": {
        "url": "https://ai.x39.org/asiawomen/beer.safetensors",
        "repo": LORA_DIR,
        "weights": "beer.safetensors",
        "adapter_name": "beer",
    },
    "Shaved": {
        "url": "https://ai.x39.org/lora/shavedpussy25.safetensors",
        "repo": LORA_DIR,
        "weights": "shavedpussy25.safetensors",
        "adapter_name": "Shaved",
    },
    "CNPussy/Shaved": {
        "combo": ["CNPussy", "Shaved"],
        # "combo": ["Clitoris"],
        "adapter_weights": [0.0, 1.0],
    },
    "Anime-V2": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "weights": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
        "adapter_name": "anime-v2",
    },
    "Light-Migration": {
        "repo": "dx8152/Qwen-Edit-2509-Light-Migration",
        "weights": "参考色调.safetensors",
        "adapter_name": "light-migration",
    },
    "Upscaler": {
        "repo": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        "weights": "qwen_image_edit_2511_upscale.safetensors",
        "adapter_name": "upscale-2k",
    },
    "Style-Transfer": {
        "repo": "zooeyy/Style-Transfer",
        "weights": "Style Transfer-Alpha-V0.1.safetensors",
        "adapter_name": "style-transfer",
    },
    "Manga-Tone": {
        "repo": "nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
        "weights": "tone001.safetensors",
        "adapter_name": "manga-tone",
    },
    "Anything2Real": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601.safetensors",
        "adapter_name": "anything2real",
    },
    "Fal-Multiple-Angles": {
        "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "weights": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "adapter_name": "fal-multiple-angles",
    },
    "Polaroid-Photo": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
        "weights": "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
        "adapter_name": "polaroid-photo",
    },
    "Unblur-Anything": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
        "weights": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
        "adapter_name": "unblur-anything",
    },
    "Midnight-Noir-Eyes-Spotlight": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
        "weights": "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
        "adapter_name": "midnight-noir-eyes-spotlight",
    },
    "Hyper-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
        "adapter_name": "hyper-realistic-portrait",
    },
    "Ultra-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
        "adapter_name": "ultra-realistic-portrait",
    },
    "Pixar-Inspired-3D": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
        "weights": "PI3_20.safetensors",
        "adapter_name": "pi3",
    },
    "Noir-Comic-Book": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
        "weights": "Noir-Comic-Book-Panel_20.safetensors",
        "adapter_name": "ncb",
    },
    "Any-light": {
        "repo": "lilylilith/QIE-2511-MP-AnyLight",
        "weights": "QIE-2511-AnyLight_.safetensors",
        "adapter_name": "any-light",
    },
    "Studio-DeLight": {
        "repo": "prithivMLmods/QIE-2511-Studio-DeLight",
        "weights": "QIE-2511-Studio-DeLight-5000.safetensors",
        "adapter_name": "studio-delight",
    },
    "Cinematic-FlatLog": {
        "repo": "prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
        "weights": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
        "adapter_name": "flat-log",
    },
}

LOADED_ADAPTERS: set = set()
CURRENT_ACTIVE_ADAPTER = None
ADAPTER_NAMES = [name for name in ADAPTER_SPECS.keys() if not name.startswith("_")]
COMBO_SPECS = {
    name: {
        "combo": spec["combo"],
        "adapter_weights": spec.get("adapter_weights", [1.0] * len(spec["combo"])),
    }
    for name, spec in ADAPTER_SPECS.items()
    if "combo" in spec
}
COMBO_SPECS_JSON = json.dumps(COMBO_SPECS, ensure_ascii=False)


def ensure_adapter_source(spec):
    if "url" not in spec:
        return spec["repo"], spec["weights"]

    target_dir = spec["repo"]
    target_name = spec["weights"]
    target_path = os.path.join(target_dir, target_name)

    if not os.path.exists(target_path):
        print(f"--- Downloading external LoRA: {spec['url']} -> {target_path} ---")
        with requests.get(spec["url"], stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"--- Downloaded external LoRA to: {target_path} ---")

    return target_dir, target_name


def ensure_single_adapter_loaded(display_name, spec):
    adapter_name = spec["adapter_name"]
    if adapter_name in LOADED_ADAPTERS:
        print(f"--- Adapter {display_name} already loaded. ---")
        return adapter_name

    print(f"--- Downloading and Loading Adapter: {display_name} ---")
    repo_or_dir, weight_name = ensure_adapter_source(spec)
    pipe.load_lora_weights(repo_or_dir, weight_name=weight_name, adapter_name=adapter_name)
    LOADED_ADAPTERS.add(adapter_name)
    return adapter_name

EXAMPLES_CONFIG = [
    {"images": ["examples/xy8.jpg"],                        "prompt": "Use these concepts in your prompts to activate specific capabilities:\nnsfw nipples vagina penis missionary cowgirlout reversecowgirlpov blowjob cum_on_face creamp1e l1ck",                                                                                           "lora": "XY"},
    {"images": ["examples/dog.jpg"],                        "prompt": "zl_cn_female_body_v1,A 20-years-old Chinese girl,standing, big breasts,short hair, background is a rural  town",                                                                                           "lora": "CNFemale"},
    {"images": ["examples/pussy.jpg"],                        "prompt": "zl_cn_female_genitals_v1, full nude, 乳房很大，跪着，腿分开, expose pussy, expose asshole, front view",                                                                                           "lora": "CNPussy"},
    {"images": ["examples/ban.jpg"],                        "prompt": "replace the woman with photo of LL9_BANYET_01, middle-aged Asian man, keep the pose, clothing and background unchanged.",                                                                                           "lora": "banyetman"},
    {"images": ["examples/ban.jpg"],                        "prompt": "zl_banyetnuts_v2, Nuts Locked Belt, front view ",                                                                                           "lora": "lock"},
    {"images": ["examples/asiawomen.jpg"],                        "prompt": "the woman doing LL9_BANYET_02 beer bottle insert to her vaginal, nude, keep the pose, clothing and background unchanged.",                                                                                           "lora": "beer"},    
    {"images": ["examples/asiawomen.jpg"],                        "prompt": "zl_cn_female_genitals_v1, zl_shavedpussy_v1, full nude, 躺着，腿分开, 头扭向侧面，低头，expose wide pussy, asshole, direct view.",                                                                                           "lora": "CNPussy/Shaved"},
    {"images": ["examples/HRP.jpg"],                        "prompt": "Transform into a hyper-realistic face portrait.",                                                                 "lora": "Hyper-Realistic-Portrait"},
    {"images": ["examples/A.jpeg"],                         "prompt": "Rotate the camera 45 degrees to the right.",                                                                      "lora": "Multiple-Angles"},
    {"images": ["examples/U.jpg"],                          "prompt": "Upscale this picture to 4K resolution.",                                                                          "lora": "Upscaler"},
    {"images": ["examples/L1.jpg", "examples/L2.jpg"],      "prompt": "Apply the lighting from image 2 to image 1.",                                                                     "lora": "Any-light"},
    {"images": ["examples/PP1.jpg"],                        "prompt": "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed preserving realistic texture and details.", "lora": "Polaroid-Photo"},
    {"images": ["examples/Z1.jpg"],                         "prompt": "Front-right quarter view.",                                                                                       "lora": "Fal-Multiple-Angles"},
    {"images": ["examples/URP.jpg"],                        "prompt": "Transform into a cinematic flat log.",                                                                            "lora": "Cinematic-FlatLog"},
    {"images": ["examples/SL.jpg"],                         "prompt": "Neutral uniform lighting. Preserve identity and composition.",                                                    "lora": "Studio-DeLight"},
    {"images": ["examples/PI.jpg"],                         "prompt": "Transform it into Pixar-inspired 3D.",                                                                            "lora": "Pixar-Inspired-3D"},
    {"images": ["examples/MT.jpg"],                         "prompt": "Paint with manga tone.",                                                                                          "lora": "Manga-Tone"},
    {"images": ["examples/NCB.jpg"],                        "prompt": "Transform into a noir comic book style.",                                                                         "lora": "Noir-Comic-Book"},
    {"images": ["examples/URP.jpg"],                        "prompt": "Ultra-realistic portrait.",                                                                                       "lora": "Ultra-Realistic-Portrait"},
    {"images": ["examples/MN.jpg"],                         "prompt": "Transform into Midnight Noir Eyes Spotlight.",                                                                    "lora": "Midnight-Noir-Eyes-Spotlight"},
    {"images": ["examples/ST1.jpg", "examples/ST2.jpg"],    "prompt": "Convert Image 1 to the style of Image 2.",                                                                        "lora": "Style-Transfer"},
    {"images": ["examples/R1.jpg"],                         "prompt": "Change the picture to realistic photograph.",                                                                     "lora": "Anything2Real"},
    {"images": ["examples/UA.jpeg"],                        "prompt": "Unblur and upscale.",                                                                                             "lora": "Unblur-Anything"},
    {"images": ["examples/L1.jpg", "examples/L2.jpg"],      "prompt": "Refer to the color tone, remove the original lighting from Image 1, and relight Image 1 based on the lighting and color tone of Image 2.", "lora": "Light-Migration"},
    {"images": ["examples/P1.jpg"],                         "prompt": "Transform into anime (while preserving the background and remaining elements maintaining realism and original details.)", "lora": "Anime-V2"},
]


def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""


def build_example_cards_html():
    cards = ""
    for i, ex in enumerate(EXAMPLES_CONFIG):
        thumbs_html = ""
        for path in ex["images"]:
            thumb = make_thumb_b64(path)
            if thumb:
                thumbs_html += f'<img src="{thumb}" alt="">'
            else:
                thumbs_html += '<div class="example-thumb-placeholder">Preview</div>'
        n = len(ex["images"])
        img_badge = f'{n} image{"s" if n > 1 else ""}'
        lora_badge = html_lib.escape(ex["lora"])
        prompt_short = html_lib.escape(ex["prompt"][:85])
        if len(ex["prompt"]) > 85:
            prompt_short += "…"
        cards += f'''<div class="example-card" data-idx="{i}">
            <div class="example-thumbs">{thumbs_html}</div>
            <div class="example-meta">
                <span class="example-badge">{img_badge}</span>
                <span class="example-lora-badge">{lora_badge}</span>
            </div>
            <div class="example-prompt-text">{prompt_short}</div>
        </div>'''
    return cards


def load_example_data(idx_str):
    try:
        idx = int(float(idx_str)) if idx_str and idx_str.strip() else -1
    except (ValueError, TypeError):
        idx = -1
    if idx < 0 or idx >= len(EXAMPLES_CONFIG):
        return json.dumps({"images": [], "prompt": "", "lora": "", "names": [], "status": "error"})
    ex = EXAMPLES_CONFIG[idx]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return json.dumps({"images": b64_list, "prompt": ex["prompt"], "lora": ex["lora"], "names": names, "status": "ok"})


print("Building example thumbnails…")
EXAMPLE_CARDS_HTML = build_example_cards_html()
print(f"Built {len(EXAMPLES_CONFIG)} example cards.")


def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images


def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    w, h = image.size
    if w > h:
        nw = min(w, MAX_OUTPUT_SIDE)
        nh = int(nw * h / w)
    else:
        nh = min(h, MAX_OUTPUT_SIDE)
        nw = int(nh * w / h)
    return (nw // 8) * 8, (nh // 8) * 8


def get_dimensions_from_preset(image, resolution_preset):
    preset = (resolution_preset or "Auto").strip()
    if preset == "iPhone 15 Pro Portrait":
        return 944, 2048
    if preset == "1080p Portrait":
        return 1080, 1920
    if preset == "1080p Landscape":
        return 1920, 1080
    if preset == "Max Portrait (1152x2048)":
        return 1152, 2048
    if preset == "Max Landscape (2048x1152)":
        return 2048, 1152
    return update_dimensions_on_upload(image)


def get_next_prompt_path():
    max_index = 0
    try:
        for name in os.listdir(PROMPT_SAVE_DIR):
            if len(name) == 9 and name.startswith("p") and name.endswith(".txt"):
                num = name[1:5]
                if num.isdigit():
                    max_index = max(max_index, int(num))
    except FileNotFoundError:
        os.makedirs(PROMPT_SAVE_DIR, exist_ok=True)
    return os.path.join(PROMPT_SAVE_DIR, f"p{max_index + 1:04d}.txt")


@spaces.GPU(size="xlarge")
def infer(
    images_b64_json,
    prompt,
    lora_adapter,
    resolution_preset,
    combo_weights_json,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    batch_count,
    progress=gr.Progress(track_tqdm=True),
):
    global CURRENT_ACTIVE_ADAPTER
    print("infer() called")
    print("lora_adapter =", lora_adapter)
    print("prompt =", prompt)
    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image to edit.")
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter an edit prompt.")
    prompt = prompt.strip()

    prompt_save_path = get_next_prompt_path()
    with open(prompt_save_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print("Saved prompt to:", prompt_save_path)

    spec = ADAPTER_SPECS.get(lora_adapter)
    if not spec:
        raise gr.Error(f"Configuration not found for: {lora_adapter}")

    try:
        if "combo" in spec:
            adapter_names = []
            combo_labels = []
            for combo_key in spec["combo"]:
                combo_spec = ADAPTER_SPECS.get(combo_key)
                if not combo_spec or "adapter_name" not in combo_spec:
                    raise gr.Error(f"Configuration not found for combo adapter: {combo_key}")
                adapter_names.append(ensure_single_adapter_loaded(combo_key, combo_spec))
                combo_labels.append(combo_key)
            adapter_weights = spec.get("adapter_weights", [1.0] * len(adapter_names))
            if combo_weights_json and str(combo_weights_json).strip():
                try:
                    combo_weights = json.loads(combo_weights_json)
                    if isinstance(combo_weights, list):
                        adapter_weights = [
                            float(combo_weights[i]) if i < len(combo_weights) else float(adapter_weights[i])
                            for i in range(len(adapter_names))
                        ]
                except Exception:
                    pass
            active_signature = (tuple(adapter_names), tuple(float(w) for w in adapter_weights))
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            CURRENT_ACTIVE_ADAPTER = active_signature
            print(f"--- Activated combo adapter: {' + '.join(combo_labels)} with weights {adapter_weights} ---")
        else:
            adapter_name = ensure_single_adapter_loaded(lora_adapter, spec)
            if CURRENT_ACTIVE_ADAPTER != adapter_name:
                pipe.set_adapters([adapter_name], adapter_weights=[1.0])
                CURRENT_ACTIVE_ADAPTER = adapter_name
                print(f"--- Activated adapter: {lora_adapter} ---")
            else:
                print(f"--- Adapter {lora_adapter} already active. ---")
    except Exception as e:
        raise gr.Error(f"Failed to load adapter {lora_adapter}: {e}")

    try:
        batch_count = int(batch_count)
    except Exception:
        batch_count = 1
    batch_count = max(1, batch_count)

    try:
        seed = int(seed)
    except Exception:
        seed = 0

    negative_prompt = (
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
    )
    width, height = get_dimensions_from_preset(pil_images[0], resolution_preset)

    try:
        from datetime import datetime

        saved_paths = []
        last_preview = None
        last_seed = seed
        last_out_path = ""
        batch_zip_path = ""
        progress_text = f"0/{batch_count}"

        for idx in range(batch_count):
            current_seed = random.randint(0, MAX_SEED) if randomize_seed else seed + idx
            generator = torch.Generator(device=device).manual_seed(current_seed)
            progress((idx, batch_count), desc=f"Generating {idx + 1}/{batch_count}")

            result_image = pipe(
                image=pil_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator,
                true_cfg_scale=guidance_scale,
            ).images[0]

            out_name = f"qwen_edit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{current_seed}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            result_image.save(out_path)
            print(f"Saved to: {out_path}")

            preview_image = result_image.copy()
            preview_image.thumbnail((1280, 1280), LANCZOS)

            saved_paths.append(out_path)
            last_preview = preview_image
            last_seed = current_seed
            last_out_path = out_path
            progress_text = f"{idx + 1}/{batch_count}"

            yield last_preview, last_seed, last_out_path, "", progress_text

        if batch_count > 1 and saved_paths:
            batch_zip_name = f"qwen_batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.zip"
            batch_zip_path = os.path.join(OUTPUT_DIR, batch_zip_name)
            with zipfile.ZipFile(batch_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for saved_path in saved_paths:
                    zf.write(saved_path, arcname=os.path.basename(saved_path))
                if os.path.exists(prompt_save_path):
                    zf.write(prompt_save_path, arcname=os.path.basename(prompt_save_path))
            print(f"Batch zip saved to: {batch_zip_path}")

        yield last_preview, last_seed, last_out_path, batch_zip_path, progress_text
    except Exception as e:
        raise e


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*{box-sizing:border-box;margin:0;padding:0}

:root{
    --or:       #FF4500;
    --or-dim:   #CC3700;
    --or-bright:#FF6A33;
    --or-glow:  rgba(255,69,0,.28);
    --or-soft:  rgba(255,69,0,.12);
    --or-xsoft: rgba(255,69,0,.06);
    --bg:       #0d0d0f;
    --bg1:      #141416;
    --bg2:      #1a1a1d;
    --bg3:      #222226;
    --border:   #2a2a2e;
    --border2:  #333338;
    --text:     #e8e8ec;
    --text2:    #9898a8;
    --text3:    #555560;
    --mono:     'JetBrains Mono', monospace;
}

html, body, .gradio-container {
    color-scheme: dark !important;
    background: #0d0d0f !important;
}

body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 14px !important;
    color: #e8e8ec !important;
    min-height: 100vh;
    background: #0d0d0f !important;
}

footer { display: none !important; }
.hidden-input { display:none!important;height:0!important;overflow:hidden!important;margin:0!important;padding:0!important; }

#example-load-btn, #gradio-run-btn {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: .01 !important;
    pointer-events: none !important;
    overflow: hidden !important;
}

/* ── App shell ─────────────────────────────────────────────────────────────── */
.app-shell {
    background: #141416 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 18px;
    margin: 12px auto;
    max-width: 1440px;
    overflow: hidden;
    box-shadow:
        0 32px 64px -16px rgba(0,0,0,.8),
        0 0 0 1px rgba(255,255,255,.03),
        0 0 60px -20px rgba(255,69,0,.2);
}

/* ── Header ────────────────────────────────────────────────────────────────── */
.app-header {
    background: #1a1a1d !important;
    border-bottom: 1px solid #2a2a2e !important;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.app-header-left { display:flex; align-items:center; gap:12px; }
.app-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #FF4500, #FF6A33);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px rgba(255,69,0,.3);
    flex-shrink: 0;
}
.app-logo svg { width:22px; height:22px; fill:#fff; }
.app-title {
    font-size: 18px; font-weight: 800; letter-spacing: -.4px;
    background: linear-gradient(135deg, #fff, #aaa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Inter', sans-serif;
}
.app-badge {
    font-size: 11px; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; letter-spacing: .3px;
    background: rgba(255,69,0,.12);
    color: #FF6A33 !important;
    -webkit-text-fill-color: #FF6A33 !important;
    border: 1px solid rgba(255,69,0,.3);
    font-family: 'Inter', sans-serif;
}
.app-badge.fast {
    background: rgba(34,197,94,.1);
    color: #4ade80 !important;
    -webkit-text-fill-color: #4ade80 !important;
    border: 1px solid rgba(34,197,94,.25);
}

/* ── GitHub button - highlighted, always same in light & dark ────────────── */
.gh-btn {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    padding: 7px 14px !important;
    border-radius: 8px !important;
    text-decoration: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: .1px !important;
    /* Fixed colors - never change with theme */
    background: #FF4500 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255,255,255,.15) !important;
    box-shadow:
        0 2px 8px rgba(255,69,0,.4),
        0 1px 0 rgba(255,255,255,.1) inset !important;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
}
.gh-btn:hover {
    background: #FF6A33 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow:
        0 4px 16px rgba(255,69,0,.55),
        0 1px 0 rgba(255,255,255,.12) inset !important;
}
.gh-btn:active {
    background: #CC3700 !important;
    transform: translateY(0) !important;
    box-shadow: 0 1px 4px rgba(255,69,0,.3) !important;
}
.gh-btn svg {
    fill: #ffffff !important;
    flex-shrink: 0;
    width: 15px !important;
    height: 15px !important;
}
.gh-btn span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Lock GitHub button against ANY browser/OS theme override */
@media (prefers-color-scheme: light) {
    .gh-btn {
        background: #FF4500 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border-color: rgba(255,255,255,.15) !important;
    }
    .gh-btn:hover {
        background: #FF6A33 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .gh-btn svg { fill: #ffffff !important; }
    .gh-btn span { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
}
@media (prefers-color-scheme: dark) {
    .gh-btn {
        background: #FF4500 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .gh-btn:hover { background: #FF6A33 !important; }
    .gh-btn svg { fill: #ffffff !important; }
    .gh-btn span { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
}

/* ── Toolbar ───────────────────────────────────────────────────────────────── */
.app-toolbar {
    background: #141416 !important;
    border-bottom: 1px solid #2a2a2e !important;
    padding: 7px 16px;
    display: flex; gap: 4px; align-items: center; flex-wrap: wrap;
}
.tb-sep { width:1px; height:28px; background:#2a2a2e; margin:0 8px; }
.modern-tb-btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    min-width: 32px; height: 34px;
    background: transparent; border: 1px solid transparent;
    border-radius: 8px; cursor: pointer;
    font-size: 13px; font-weight: 700; padding: 0 12px;
    font-family: 'Inter', sans-serif;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    transition: all .15s ease;
}
.modern-tb-btn:hover { background: rgba(255,69,0,.12); border-color: rgba(255,69,0,.4); }
.modern-tb-btn:active { background: rgba(255,69,0,.22); border-color: #FF4500; }
.modern-tb-btn .tb-label { font-size:13px; color:#fff!important; -webkit-text-fill-color:#fff!important; font-weight:700; }
.modern-tb-btn .tb-svg { width:15px; height:15px; flex-shrink:0; }
.modern-tb-btn .tb-svg, .modern-tb-btn .tb-svg * { stroke:#fff!important; fill:none!important; }
.tb-info { font-family:var(--mono); font-size:12px; color:#555560; padding:0 8px; display:flex; align-items:center; }

@media (prefers-color-scheme: light) {
    .modern-tb-btn { color:#fff!important; -webkit-text-fill-color:#fff!important; }
    .modern-tb-btn .tb-label { color:#fff!important; -webkit-text-fill-color:#fff!important; }
    .modern-tb-btn .tb-svg, .modern-tb-btn .tb-svg * { stroke:#fff!important; }
}

/* ── Main layout ───────────────────────────────────────────────────────────── */
.app-main-row { display:flex; gap:0; flex:1; overflow:hidden; }
.app-main-left { flex:1; display:flex; flex-direction:column; min-width:0; border-right:1px solid #2a2a2e; }
.app-main-right { width:440px; display:flex; flex-direction:column; flex-shrink:0; background:#141416!important; }

/* ── Drop zone ─────────────────────────────────────────────────────────────── */
#gallery-drop-zone { position:relative; background:#09090b!important; min-height:220px; overflow:auto; padding:14px 0; box-sizing:border-box; }
#gallery-drop-zone.drag-over { outline:2px solid #FF4500; outline-offset:-2px; background:rgba(255,69,0,.05)!important; }

.upload-prompt-modern { position:absolute; top:50%; left:24px; right:24px; transform:translateY(-50%); z-index:20; }
.upload-click-area {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    cursor:pointer; padding:28px 52px; border:2px dashed #333338; border-radius:16px;
    background:rgba(255,69,0,.05); transition:all .2s ease; gap:8px;
}
.upload-click-area:hover { background:rgba(255,69,0,.1); border-color:#FF4500; transform:scale(1.03); }
.upload-click-area:active { background:rgba(255,69,0,.15); transform:scale(.98); }
.upload-click-area svg { width:80px; height:80px; }
.upload-main-text { color:#9898a8; font-size:14px; font-weight:600; margin-top:4px; font-family:'Inter',sans-serif; }
.upload-sub-text { color:#555560; font-size:12px; font-weight:400; text-align:center; max-width:280px; line-height:1.5; font-family:'Inter',sans-serif; }

/* ── Gallery grid ──────────────────────────────────────────────────────────── */
.image-gallery-grid {
    display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
    gap:12px; padding:16px; align-content:start;
}
.gallery-thumb {
    position:relative; aspect-ratio:1; border-radius:10px; overflow:hidden;
    cursor:pointer; border:2px solid #2a2a2e; transition:all .2s ease; background:#1a1a1d;
}
.gallery-thumb:hover { border-color:#333338; transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,.4); }
.gallery-thumb.selected { border-color:#FF4500!important; box-shadow:0 0 0 3px rgba(255,69,0,.25); }
.gallery-thumb img { width:100%; height:100%; object-fit:cover; }
.thumb-badge {
    position:absolute; top:6px; left:6px; background:#FF4500; color:#fff;
    padding:2px 8px; border-radius:4px; font-family:var(--mono); font-size:11px; font-weight:600;
}
.thumb-remove {
    position:absolute; top:6px; right:6px; width:24px; height:24px;
    background:rgba(0,0,0,.75); color:#fff; border:1px solid rgba(255,255,255,.15);
    border-radius:50%; cursor:pointer; display:none;
    align-items:center; justify-content:center; font-size:12px; transition:all .15s; line-height:1;
}
.gallery-thumb:hover .thumb-remove { display:flex; }
.thumb-remove:hover { background:#FF4500; border-color:#FF4500; }
.gallery-add-card {
    aspect-ratio:1; border-radius:10px; border:2px dashed #333338;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    cursor:pointer; transition:all .2s ease; background:rgba(255,69,0,.05); gap:4px;
}
.gallery-add-card:hover { border-color:#FF4500; background:rgba(255,69,0,.1); }
.gallery-add-card .add-icon { font-size:28px; color:#555560; font-weight:300; }
.gallery-add-card .add-text { font-size:12px; color:#555560; font-weight:600; font-family:'Inter',sans-serif; }

/* ── Hint bar ──────────────────────────────────────────────────────────────── */
.hint-bar {
    background: rgba(255,69,0,.05) !important;
    border-top:1px solid #2a2a2e; border-bottom:1px solid #2a2a2e;
    padding:10px 20px; font-size:13px; color:#9898a8; line-height:1.7;
    font-weight:400; font-family:'Inter',sans-serif;
}
.hint-bar b { color:#FF7A4D; font-weight:700; }
.hint-bar kbd {
    display:inline-block; padding:1px 6px; background:#222226;
    border:1px solid #333338; border-radius:4px;
    font-family:var(--mono); font-size:11px; color:#9898a8;
}

/* ── Suggestions ───────────────────────────────────────────────────────────── */
.suggestions-section { border-top:1px solid #2a2a2e; padding:12px 16px; background:#141416!important; }
.suggestions-title, .examples-title {
    font-size:11px; font-weight:700; color:#555560;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;
    font-family:'Inter',sans-serif;
}
.suggestions-wrap { display:flex; flex-wrap:wrap; gap:6px; }
.suggestion-chip {
    display:inline-flex; align-items:center; gap:4px; padding:5px 13px;
    background:rgba(255,69,0,.1); border:1px solid rgba(255,69,0,.25); border-radius:20px;
    color:#FF7A4D; font-size:12px; font-weight:600; font-family:'Inter',sans-serif;
    cursor:pointer; transition:all .15s; white-space:nowrap;
}
.suggestion-chip:hover { background:rgba(255,69,0,.2); border-color:#FF4500; color:#fff; transform:translateY(-1px); }

/* ── Examples ──────────────────────────────────────────────────────────────── */
.examples-section { border-top:1px solid #2a2a2e; padding:14px 16px 18px; background:#141416!important; width:100%; }
.examples-scroll { display:flex; gap:10px; overflow-x:auto; padding-bottom:10px; padding-top:2px; }
.examples-scroll::-webkit-scrollbar { height:5px; }
.examples-scroll::-webkit-scrollbar-track { background:#0d0d0f; border-radius:3px; }
.examples-scroll::-webkit-scrollbar-thumb { background:#333338; border-radius:3px; }
.examples-scroll::-webkit-scrollbar-thumb:hover { background:#CC3700; }
.example-card {
    flex-shrink:0; width:220px; background:#1a1a1d!important; border:1px solid #2a2a2e;
    border-radius:12px; overflow:hidden; cursor:pointer; transition:all .2s ease;
}
.example-card:hover { border-color:#FF4500; transform:translateY(-3px); box-shadow:0 6px 20px rgba(255,69,0,.22); }
.example-card.loading { opacity:.5; pointer-events:none; }
.example-thumbs { display:flex; height:115px; overflow:hidden; background:#222226!important; }
.example-thumbs img { flex:1; object-fit:cover; min-width:0; }
.example-thumb-placeholder {
    flex:1; display:flex; align-items:center; justify-content:center;
    background:#222226!important; color:#555560; font-size:11px; min-width:0;
}
.example-meta { padding:7px 10px 3px; display:flex; align-items:center; gap:5px; flex-wrap:wrap; }
.example-badge {
    display:inline-flex; padding:2px 7px; background:rgba(255,69,0,.1); border-radius:4px;
    font-size:10px; font-weight:700; color:#FF6A33; font-family:var(--mono);
    white-space:nowrap; border:1px solid rgba(255,69,0,.2);
}
.example-lora-badge {
    display:inline-flex; padding:2px 7px; background:rgba(255,255,255,.06); border-radius:4px;
    font-size:10px; font-weight:600; color:#9898a8; font-family:var(--mono);
    white-space:nowrap; border:1px solid #2a2a2e;
    max-width:120px; overflow:hidden; text-overflow:ellipsis;
}
.example-prompt-text {
    padding:2px 10px 10px; font-size:11.5px; color:#9898a8; line-height:1.45;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
    overflow:hidden; font-weight:400; font-family:'Inter',sans-serif;
}

/* ── Right panel ───────────────────────────────────────────────────────────── */
.panel-card { border-bottom:1px solid #2a2a2e; background:#141416!important; }
.panel-card-title {
    padding:11px 20px; font-size:11px; font-weight:700; color:#555560;
    text-transform:uppercase; letter-spacing:1px;
    border-bottom:1px solid rgba(42,42,46,.6);
    background:#141416!important; font-family:'Inter',sans-serif;
}
.panel-card-body { padding:14px 18px; display:flex; flex-direction:column; gap:8px; background:#141416!important; }
.modern-label { font-size:13px; font-weight:600; color:#9898a8; margin-bottom:4px; display:block; font-family:'Inter',sans-serif; }
.modern-textarea {
    width:100%; background:#09090b!important; border:1px solid #2a2a2e; border-radius:8px;
    padding:10px 14px; font-family:'Inter',sans-serif; font-size:14px;
    color:#e8e8ec!important; -webkit-text-fill-color:#e8e8ec!important;
    resize:vertical; outline:none; min-height:88px; transition:border-color .2s; font-weight:400;
}
.modern-textarea:focus { border-color:#FF4500; box-shadow:0 0 0 3px rgba(255,69,0,.22); }
.modern-textarea::placeholder { color:#555560!important; -webkit-text-fill-color:#555560!important; }
.modern-textarea.error-flash {
    border-color:#ef4444!important;
    box-shadow:0 0 0 3px rgba(239,68,68,.2)!important;
    animation:shake .4s ease;
}
@keyframes shake {
    0%,100%{transform:translateX(0)}
    20%,60%{transform:translateX(-4px)}
    40%,80%{transform:translateX(4px)}
}

/* ── LoRA selector - always dark ───────────────────────────────────────────── */
.lora-selector-card { border-bottom:1px solid #2a2a2e!important; background:#0d0d0f!important; }
.lora-selector-body { padding:12px 18px!important; background:#0d0d0f!important; }
.lora-select-label {
    font-size:11px!important; font-weight:700!important;
    color:#555560!important; -webkit-text-fill-color:#555560!important;
    text-transform:uppercase!important; letter-spacing:1px!important;
    margin-bottom:8px!important; display:flex!important; align-items:center!important;
    gap:6px!important; font-family:'Inter',sans-serif!important;
}
.lora-select-label::before {
    content:''; display:inline-block; width:8px; height:8px;
    background:#FF4500; border-radius:50%; flex-shrink:0;
}
.lora-native-select {
    width:100%!important; background:#09090b!important;
    border:1px solid #333338!important; border-radius:8px!important;
    padding:9px 36px 9px 14px!important;
    font-family:'Inter',sans-serif!important;
    font-size:13px!important; font-weight:600!important;
    color:#e8e8ec!important; -webkit-text-fill-color:#e8e8ec!important;
    outline:none!important; appearance:none!important; -webkit-appearance:none!important;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23FF4500' d='M6 8L1 3h10z'/%3E%3C/svg%3E")!important;
    background-repeat:no-repeat!important; background-position:right 12px center!important;
    cursor:pointer!important; transition:border-color .2s!important; color-scheme:dark!important;
}
.lora-native-select:focus { border-color:#FF4500!important; box-shadow:0 0 0 3px rgba(255,69,0,.22)!important; }
.lora-native-select option { background:#1a1a1d!important; color:#e8e8ec!important; }

@media (prefers-color-scheme: light) {
    .lora-selector-card { background:#0d0d0f!important; }
    .lora-selector-body { background:#0d0d0f!important; }
    .lora-select-label { color:#555560!important; -webkit-text-fill-color:#555560!important; }
    .lora-native-select {
        background:#09090b!important; color:#e8e8ec!important;
        -webkit-text-fill-color:#e8e8ec!important;
        border-color:#333338!important; color-scheme:dark!important;
    }
    .lora-native-select option { background:#1a1a1d!important; color:#e8e8ec!important; }
}

/* ── Toast ─────────────────────────────────────────────────────────────────── */
.toast-notification {
    position:fixed; top:24px; left:50%;
    transform:translateX(-50%) translateY(-120%);
    z-index:9999; padding:10px 24px; border-radius:10px;
    font-family:'Inter',sans-serif; font-size:14px; font-weight:700;
    display:flex; align-items:center; gap:8px;
    box-shadow:0 8px 24px rgba(0,0,0,.5);
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),opacity .35s ease;
    opacity:0; pointer-events:none;
}
.toast-notification.visible { transform:translateX(-50%) translateY(0); opacity:1; pointer-events:auto; }
.toast-notification.error   { background:linear-gradient(135deg,#dc2626,#b91c1c); color:#fff; border:1px solid rgba(255,255,255,.15); }
.toast-notification.warning { background:linear-gradient(135deg,#FF4500,#CC3700); color:#fff; border:1px solid rgba(255,255,255,.15); }
.toast-notification.info    { background:linear-gradient(135deg,#2563eb,#1d4ed8); color:#fff; border:1px solid rgba(255,255,255,.15); }

/* ── Run button ────────────────────────────────────────────────────────────── */
.btn-run {
    display:flex; align-items:center; justify-content:center; gap:8px; width:100%;
    background:linear-gradient(135deg,#FF4500,#CC3700); border:none; border-radius:10px;
    padding:13px 24px; cursor:pointer; font-size:15px; font-weight:800;
    font-family:'Inter',sans-serif;
    color:#fff!important; -webkit-text-fill-color:#fff!important;
    transition:all .2s ease; letter-spacing:.2px;
    box-shadow:0 4px 20px rgba(255,69,0,.28),inset 0 1px 0 rgba(255,255,255,.12);
}
.btn-run:hover {
    background:linear-gradient(135deg,#FF6A33,#FF4500); transform:translateY(-1px);
    box-shadow:0 8px 28px rgba(255,69,0,.45),inset 0 1px 0 rgba(255,255,255,.15);
}
.btn-run:active { transform:translateY(0); box-shadow:0 2px 10px rgba(255,69,0,.28); }
.btn-run svg { width:18px; height:18px; fill:#fff!important; }
#custom-run-btn, #custom-run-btn *, #run-btn-label, .btn-run, .btn-run * {
    color:#fff!important; -webkit-text-fill-color:#fff!important; fill:#fff!important;
}

/* ── Output ────────────────────────────────────────────────────────────────── */
.output-frame { border-bottom:1px solid #2a2a2e; display:flex; flex-direction:column; position:relative; }
.out-title {
    padding:10px 20px; font-size:11px; font-weight:700;
    color:#fff!important; -webkit-text-fill-color:#fff!important;
    text-transform:uppercase; letter-spacing:1px;
    border-bottom:1px solid rgba(42,42,46,.6);
    display:flex; align-items:center; justify-content:space-between;
    background:#141416!important; font-family:'Inter',sans-serif;
}
.out-body {
    flex:1; background:#09090b!important;
    display:flex; align-items:center; justify-content:center;
    overflow:hidden; min-height:180px; position:relative;
}
.out-body img { width:100%; height:100%; object-fit:cover; image-rendering:auto; display:block; }
.out-placeholder { color:#555560; font-size:13px; text-align:center; padding:20px; font-weight:500; font-family:'Inter',sans-serif; }
.out-download-btn {
    display:none; align-items:center; justify-content:center;
    background:rgba(255,69,0,.12); border:1px solid rgba(255,69,0,.28); border-radius:6px;
    cursor:pointer; padding:3px 10px; font-size:11px; font-weight:700;
    color:#FF6A33!important; -webkit-text-fill-color:#FF6A33!important;
    gap:4px; height:24px; transition:all .15s; font-family:'Inter',sans-serif;
}
.out-download-btn:hover {
    background:#FF4500; border-color:#FF4500;
    color:#fff!important; -webkit-text-fill-color:#fff!important;
}
.out-download-btn.visible { display:inline-flex; }
.out-download-btn svg { width:12px; height:12px; fill:#FF6A33; }
.out-download-btn:hover svg { fill:#fff; }

/* ── Loader ────────────────────────────────────────────────────────────────── */
.modern-loader {
    display:none; position:absolute; top:0; left:0; right:0; bottom:0;
    background:rgba(9,9,11,.93); z-index:15;
    flex-direction:column; align-items:center; justify-content:center;
    gap:16px; backdrop-filter:blur(4px);
}
.modern-loader.active { display:flex; }
.modern-loader .loader-spinner {
    width:36px; height:36px; border:3px solid #2a2a2e;
    border-top-color:#FF4500; border-radius:50%; animation:spin .8s linear infinite;
}
@keyframes spin { to{transform:rotate(360deg)} }
.modern-loader .loader-text { font-size:13px; color:#9898a8; font-weight:600; font-family:'Inter',sans-serif; }
.loader-bar-track { width:200px; height:4px; background:#2a2a2e; border-radius:2px; overflow:hidden; }
.loader-bar-fill {
    height:100%; background:linear-gradient(90deg,#FF4500,#FF6A33,#FF4500);
    background-size:200% 100%; animation:shimmer 1.5s ease-in-out infinite; border-radius:2px;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* ── Settings ──────────────────────────────────────────────────────────────── */
.settings-group { border:1px solid #2a2a2e; border-radius:10px; margin:12px 16px; padding:0; overflow:hidden; background:#141416!important; }
.settings-group-title {
    font-size:11px; font-weight:700; color:#555560; text-transform:uppercase; letter-spacing:1px;
    padding:9px 16px; border-bottom:1px solid #2a2a2e;
    background:rgba(26,26,29,.5)!important; font-family:'Inter',sans-serif;
}
.settings-group-body { padding:14px 16px; display:flex; flex-direction:column; gap:12px; background:#141416!important; }
.combo-weights-panel {
    display:none;
    margin-bottom:12px;
    padding:12px 12px 4px;
    border:1px solid #2a2a2e;
    border-radius:8px;
    background:#111113!important;
}
.combo-weights-panel.visible { display:block; }
.combo-weights-title {
    font-size:11px;
    font-weight:700;
    color:#555560;
    text-transform:uppercase;
    letter-spacing:1px;
    margin-bottom:10px;
    font-family:'Inter',sans-serif;
}
.combo-weight-row {
    display:flex;
    align-items:center;
    gap:10px;
    min-height:28px;
    margin-bottom:10px;
}
.combo-weight-row label {
    font-size:13px;
    font-weight:600;
    color:#9898a8;
    min-width:92px;
    flex-shrink:0;
    font-family:'Inter',sans-serif;
}
.combo-weight-row input[type="range"] {
    flex:1;
    height:5px;
    background:#333338;
    border-radius:3px;
    outline:none;
    min-width:0;
}
.slider-row { display:flex; align-items:center; gap:10px; min-height:28px; }
.slider-row label { font-size:13px; font-weight:600; color:#9898a8; min-width:72px; flex-shrink:0; font-family:'Inter',sans-serif; }
.slider-row input[type="range"] {
    flex:1; -webkit-appearance:none; appearance:none;
    height:5px; background:#333338; border-radius:3px; outline:none; min-width:0;
}
.slider-row input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance:none; width:16px; height:16px;
    background:linear-gradient(135deg,#FF4500,#CC3700);
    border-radius:50%; cursor:pointer;
    box-shadow:0 2px 6px rgba(255,69,0,.28); transition:transform .15s;
}
.slider-row input[type="range"]::-webkit-slider-thumb:hover { transform:scale(1.2); }
.slider-row input[type="range"]::-moz-range-thumb {
    width:16px; height:16px; background:linear-gradient(135deg,#FF4500,#CC3700);
    border-radius:50%; cursor:pointer; border:none; box-shadow:0 2px 6px rgba(255,69,0,.28);
}
.slider-row .slider-val {
    min-width:52px; text-align:right; font-family:var(--mono); font-size:12px;
    font-weight:500; padding:3px 8px; background:#09090b; border:1px solid #2a2a2e;
    border-radius:6px; color:#9898a8; flex-shrink:0;
}
.checkbox-row { display:flex; align-items:center; gap:8px; font-size:13px; color:#9898a8; }
.checkbox-row input[type="checkbox"] { accent-color:#FF4500; width:16px; height:16px; cursor:pointer; }
.checkbox-row label { color:#9898a8; font-size:13px; cursor:pointer; font-weight:500; font-family:'Inter',sans-serif; }

/* ── Status bar ────────────────────────────────────────────────────────────── */
.app-statusbar {
    background:#141416!important; border-top:1px solid #2a2a2e;
    padding:6px 20px; display:flex; gap:12px; height:34px; align-items:center; font-size:12px;
}
.app-statusbar .sb-section {
    padding:0 12px; flex:1; display:flex; align-items:center;
    font-family:var(--mono); font-size:12px; color:#555560;
    overflow:hidden; white-space:nowrap;
}
.app-statusbar .sb-section.sb-fixed {
    flex:0 0 auto; min-width:90px; text-align:center; justify-content:center;
    padding:3px 12px; background:rgba(255,69,0,.1); border-radius:6px;
    color:#FF6A33; font-weight:700; border:1px solid rgba(255,69,0,.2);
}

/* ── Footer ────────────────────────────────────────────────────────────────── */
.exp-note {
    padding:10px 20px; font-size:12px; color:#555560;
    border-top:1px solid #2a2a2e; text-align:center; font-weight:500;
    background:#141416!important; font-family:'Inter',sans-serif;
}
.exp-note a { color:#FF6A33!important; -webkit-text-fill-color:#FF6A33!important; text-decoration:none; }
.exp-note a:hover { text-decoration:underline; color:#fff!important; -webkit-text-fill-color:#fff!important; }

/* ── Scrollbars ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:7px; height:7px; }
::-webkit-scrollbar-track { background:#09090b; }
::-webkit-scrollbar-thumb { background:#333338; border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:#CC3700; }

/* ── Responsive ────────────────────────────────────────────────────────────── */
@media(max-width:860px){
    .app-main-row { flex-direction:column; }
    .app-main-right { width:100%; }
    .app-main-left { border-right:none; border-bottom:1px solid #2a2a2e; }
}
"""

gallery_js = r"""
() => {
function init(attempt = 0) {
    if (window.__qwenInitDone) return;

    const galleryGrid  = document.getElementById('image-gallery-grid');
    const dropZone     = document.getElementById('gallery-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick  = document.getElementById('upload-click-area');
    const fileInput    = document.getElementById('custom-file-input');
    const btnUpload    = document.getElementById('tb-upload');
    const btnRemove    = document.getElementById('tb-remove');
    const btnClear     = document.getElementById('tb-clear');
    const promptInput  = document.getElementById('custom-prompt-input');
    const loraSelect   = document.getElementById('custom-lora-select');
    const comboWeightsPanel = document.getElementById('combo-weights-panel');
    const comboWeightsBody  = document.getElementById('combo-weights-body');
    const runBtnEl     = document.getElementById('custom-run-btn');
    const imgCountTb   = document.getElementById('tb-image-count');
    const imgCountSb   = document.getElementById('sb-image-count');

    if (!galleryGrid || !fileInput || !dropZone) {
        if (attempt < 40) setTimeout(() => init(attempt + 1), 500);
        return;
    }
    window.__qwenInitDone = true;
    const STATE_STORAGE_KEY = 'qwen_edit_ui_state_v1';
    const comboSpecs = """ + COMBO_SPECS_JSON + r""";

    let images = [];
    window.__uploadedImages = images;
    let selectedIdx = -1;
    let toastTimer  = null;

    /* ── Force dark styles on elements that browsers may override ── */
    function enforceDarkStyles() {
        /* LoRA card */
        const loraCard  = document.querySelector('.lora-selector-card');
        const loraBody  = document.querySelector('.lora-selector-body');
        const loraLabel = document.querySelector('.lora-select-label');
        const loraEl    = document.getElementById('custom-lora-select');
        if (loraCard)  { loraCard.style.setProperty('background','#0d0d0f','important'); }
        if (loraBody)  { loraBody.style.setProperty('background','#0d0d0f','important'); }
        if (loraLabel) {
            loraLabel.style.setProperty('color','#555560','important');
            loraLabel.style.setProperty('-webkit-text-fill-color','#555560','important');
        }
        if (loraEl) {
            loraEl.style.setProperty('background-color','#09090b','important');
            loraEl.style.setProperty('color','#e8e8ec','important');
            loraEl.style.setProperty('-webkit-text-fill-color','#e8e8ec','important');
            loraEl.style.setProperty('border-color','#333338','important');
        }

        /* GitHub button */
        const ghBtn = document.querySelector('.gh-btn');
        if (ghBtn) {
            ghBtn.style.setProperty('background','#FF4500','important');
            ghBtn.style.setProperty('color','#ffffff','important');
            ghBtn.style.setProperty('-webkit-text-fill-color','#ffffff','important');
            ghBtn.style.setProperty('border-color','rgba(255,255,255,.15)','important');
            ghBtn.style.setProperty('box-shadow','0 2px 8px rgba(255,69,0,.4)','important');
            const svg = ghBtn.querySelector('svg');
            if (svg) svg.style.setProperty('fill','#ffffff','important');
            const span = ghBtn.querySelector('span');
            if (span) {
                span.style.setProperty('color','#ffffff','important');
                span.style.setProperty('-webkit-text-fill-color','#ffffff','important');
            }
        }

        /* Shell sections */
        const shell   = document.querySelector('.app-shell');
        const header  = document.querySelector('.app-header');
        const toolbar = document.querySelector('.app-toolbar');
        if (shell)   shell.style.setProperty('background','#141416','important');
        if (header)  header.style.setProperty('background','#1a1a1d','important');
        if (toolbar) toolbar.style.setProperty('background','#141416','important');
    }

    enforceDarkStyles();

    /* GitHub button hover interaction */
    const ghBtn = document.querySelector('.gh-btn');
    if (ghBtn) {
        ghBtn.addEventListener('mouseenter', () => {
            ghBtn.style.setProperty('background','#FF6A33','important');
            ghBtn.style.setProperty('color','#ffffff','important');
            ghBtn.style.setProperty('-webkit-text-fill-color','#ffffff','important');
            ghBtn.style.setProperty('transform','translateY(-1px)','important');
            ghBtn.style.setProperty('box-shadow','0 4px 16px rgba(255,69,0,.55)','important');
        });
        ghBtn.addEventListener('mouseleave', () => {
            ghBtn.style.setProperty('background','#FF4500','important');
            ghBtn.style.setProperty('color','#ffffff','important');
            ghBtn.style.setProperty('-webkit-text-fill-color','#ffffff','important');
            ghBtn.style.setProperty('transform','translateY(0)','important');
            ghBtn.style.setProperty('box-shadow','0 2px 8px rgba(255,69,0,.4)','important');
        });
        ghBtn.addEventListener('mousedown', () => {
            ghBtn.style.setProperty('background','#CC3700','important');
            ghBtn.style.setProperty('transform','translateY(0)','important');
        });
        ghBtn.addEventListener('mouseup', () => {
            ghBtn.style.setProperty('background','#FF6A33','important');
        });
    }

    function showToast(message, type) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.className = 'toast-notification';
            toast.innerHTML = '<span class="toast-icon"></span><span class="toast-text"></span>';
            document.body.appendChild(toast);
        }
        const icon = toast.querySelector('.toast-icon');
        const text = toast.querySelector('.toast-text');
        toast.className = 'toast-notification ' + (type || 'error');
        icon.textContent = type === 'warning' ? '\u26A0' : type === 'info' ? '\u2139' : '\u2717';
        text.textContent = message;
        if (toastTimer) clearTimeout(toastTimer);
        void toast.offsetWidth;
        toast.classList.add('visible');
        toastTimer = setTimeout(() => toast.classList.remove('visible'), 3500);
    }
    window.__showToast = showToast;

    function flashPromptError() {
        if (!promptInput) return;
        promptInput.classList.add('error-flash');
        promptInput.focus();
        setTimeout(() => promptInput.classList.remove('error-flash'), 800);
    }

    function setGradioValue(containerId, value) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.querySelectorAll('input, textarea').forEach(el => {
            if (el.type === 'file' || el.type === 'range' || el.type === 'checkbox') return;
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) {
                ns.set.call(el, value);
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
    }
    window.__setGradioValue = setGradioValue;

    function readUiState() {
        try {
            const raw = localStorage.getItem(STATE_STORAGE_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            console.warn('Failed to read UI state:', e);
            return null;
        }
    }

    function saveUiState() {
        try {
            const seedEl = document.getElementById('custom-seed');
            const stepsEl = document.getElementById('custom-steps');
            const batchEl = document.getElementById('custom-batch');
            const comboWeights = Array.from(document.querySelectorAll('.combo-weight-slider')).map(el => el.value);
            const state = {
                images,
                prompt: promptInput ? promptInput.value : '',
                lora: loraSelect ? loraSelect.value : '',
                seed: seedEl ? seedEl.value : '0',
                steps: stepsEl ? stepsEl.value : '4',
                batch: batchEl ? batchEl.value : '1',
                comboWeights,
            };
            localStorage.setItem(STATE_STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            console.warn('Failed to save UI state:', e);
        }
    }

    function syncComboWeightsToGradio() {
        const values = Array.from(document.querySelectorAll('.combo-weight-slider')).map(el => parseFloat(el.value || '1'));
        setGradioValue('gradio-combo-weights', values.length ? JSON.stringify(values) : '');
        saveUiState();
    }

    function renderComboWeights(savedValues) {
        if (!comboWeightsPanel || !comboWeightsBody || !loraSelect) return;
        const comboSpec = comboSpecs[loraSelect.value];
        console.log('renderComboWeights', loraSelect.value, comboSpec);
        if (!comboSpec || !Array.isArray(comboSpec.combo) || comboSpec.combo.length === 0) {
            comboWeightsPanel.classList.remove('visible');
            comboWeightsBody.innerHTML = '';
            setGradioValue('gradio-combo-weights', '');
            return;
        }

        const defaults = Array.isArray(comboSpec.adapter_weights) ? comboSpec.adapter_weights : comboSpec.combo.map(() => 1.0);
        const currentValues = Array.from(comboWeightsBody.querySelectorAll('.combo-weight-slider')).map(el => el.value);
        const effectiveValues =
            Array.isArray(savedValues) && savedValues.length
                ? savedValues
                : (currentValues.length === comboSpec.combo.length ? currentValues : defaults);
        comboWeightsPanel.classList.add('visible');
        comboWeightsBody.innerHTML = comboSpec.combo.map((name, idx) => {
            const val = effectiveValues[idx] != null ? effectiveValues[idx] : defaults[idx];
            return `
                <div class="combo-weight-row">
                    <label>${name}</label>
                    <input type="range" class="combo-weight-slider" min="0" max="2" step="0.05" value="${val}">
                    <span class="slider-val">${val}</span>
                </div>
            `;
        }).join('');

        comboWeightsBody.querySelectorAll('.combo-weight-slider').forEach(slider => {
            slider.addEventListener('input', () => {
                const val = slider.parentElement.querySelector('.slider-val');
                if (val) val.textContent = slider.value;
                syncComboWeightsToGradio();
            });
            slider.addEventListener('change', syncComboWeightsToGradio);
        });
        syncComboWeightsToGradio();
    }

    function restoreUiState() {
        const state = readUiState();
        if (!state) return;

        if (Array.isArray(state.images) && state.images.length > 0) {
            images = state.images.map((img, idx) => ({
                id: img.id || (Date.now() + idx + Math.random()),
                b64: img.b64,
                name: img.name || ('image_' + (idx + 1)),
            })).filter(img => !!img.b64);
            window.__uploadedImages = images;
        }

        if (promptInput && typeof state.prompt === 'string') {
            promptInput.value = state.prompt;
        }

        if (loraSelect && typeof state.lora === 'string' && state.lora) {
            const hasOption = Array.from(loraSelect.options).some(opt => opt.value === state.lora);
            if (hasOption) loraSelect.value = state.lora;
        }

        const seedEl = document.getElementById('custom-seed');
        const seedVal = document.getElementById('custom-seed-val');
        if (seedEl && state.seed != null) {
            seedEl.value = String(state.seed);
            if (seedVal) seedVal.textContent = String(state.seed);
        }

        const stepsEl = document.getElementById('custom-steps');
        const stepsVal = document.getElementById('custom-steps-val');
        if (stepsEl && state.steps != null) {
            stepsEl.value = String(state.steps);
            if (stepsVal) stepsVal.textContent = String(state.steps);
        }

        const batchEl = document.getElementById('custom-batch');
        const batchVal = document.getElementById('custom-batch-val');
        if (batchEl && state.batch != null) {
            batchEl.value = String(state.batch);
            if (batchVal) batchVal.textContent = String(state.batch);
        }

        renderGallery();
        syncImagesToGradio();
        syncPromptToGradio();
        syncLoraToGradio();
        renderComboWeights(state.comboWeights);
        if (seedEl) seedEl.dispatchEvent(new Event('input', {bubbles:true}));
        if (stepsEl) stepsEl.dispatchEvent(new Event('input', {bubbles:true}));
        if (batchEl) batchEl.dispatchEvent(new Event('input', {bubbles:true}));
    }

    function syncImagesToGradio() {
        window.__uploadedImages = images;
        const b64Array = images.map(img => img.b64);
        setGradioValue('hidden-images-b64', JSON.stringify(b64Array));
        updateCounts();
        saveUiState();
    }
    function syncPromptToGradio() {
        if (promptInput) setGradioValue('prompt-gradio-input', promptInput.value);
        saveUiState();
    }
    function syncLoraToGradio() {
        if (!loraSelect) return;
        const preservedImages = images.map(img => ({...img}));
        const container = document.getElementById('gradio-lora');
        if (!container) return;
        container.querySelectorAll('input').forEach(el => {
            const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
            if (ns && ns.set) {
                ns.set.call(el, loraSelect.value);
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
        setTimeout(() => {
            const currentImages = Array.isArray(window.__uploadedImages) ? window.__uploadedImages : [];
            if (preservedImages.length > 0 && currentImages.length === 0) {
                images = preservedImages.map(img => ({...img}));
                window.__uploadedImages = images;
                renderGallery();
                syncImagesToGradio();
            }
        }, 250);
        renderComboWeights();
        saveUiState();
    }

    function updateCounts() {
        const n   = images.length;
        const txt = n > 0 ? n + ' image' + (n > 1 ? 's' : '') : 'No images';
        if (imgCountTb) imgCountTb.textContent = txt;
        if (imgCountSb) imgCountSb.textContent = n > 0 ? txt + ' uploaded' : 'No images uploaded';
    }

    function addImage(b64, name) {
        images.push({id: Date.now() + Math.random(), b64, name});
        renderGallery(); syncImagesToGradio();
    }
    window.__addImage = addImage;

    function removeImage(idx) {
        images.splice(idx, 1);
        if (selectedIdx === idx) selectedIdx = -1;
        else if (selectedIdx > idx) selectedIdx--;
        renderGallery(); syncImagesToGradio();
    }

    function clearAll() {
        images = []; window.__uploadedImages = images; selectedIdx = -1;
        renderGallery(); syncImagesToGradio();
    }
    window.__clearAll = clearAll;

    function renderGallery() {
        if (images.length === 0) {
            galleryGrid.innerHTML = ''; galleryGrid.style.display = 'none';
            if (uploadPrompt) uploadPrompt.style.display = '';
            return;
        }
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        galleryGrid.style.display = 'grid';
        let html = '';
        images.forEach((img, i) => {
            const sel = i === selectedIdx ? ' selected' : '';
            html += '<div class="gallery-thumb' + sel + '" data-idx="' + i + '">'
                  + '<img src="' + img.b64 + '" alt="' + (img.name||'image') + '">'
                  + '<span class="thumb-badge">#' + (i+1) + '</span>'
                  + '<button class="thumb-remove" data-remove="' + i + '">\u2715</button>'
                  + '</div>';
        });
        html += '<div class="gallery-add-card" id="gallery-add-card"><span class="add-icon">+</span><span class="add-text">Add</span></div>';
        galleryGrid.innerHTML = html;
        galleryGrid.querySelectorAll('.gallery-thumb').forEach(thumb => {
            thumb.addEventListener('click', (e) => {
                if (e.target.closest('.thumb-remove')) return;
                selectedIdx = (selectedIdx === parseInt(thumb.dataset.idx)) ? -1 : parseInt(thumb.dataset.idx);
                renderGallery();
            });
        });
        galleryGrid.querySelectorAll('.thumb-remove').forEach(btn => {
            btn.addEventListener('click', (e) => { e.stopPropagation(); removeImage(parseInt(btn.dataset.remove)); });
        });
        const addCard = document.getElementById('gallery-add-card');
        if (addCard) addCard.addEventListener('click', () => fileInput.click());
    }

    function processFiles(files) {
        Array.from(files).forEach(file => {
            if (!file.type.startsWith('image/')) return;
            const reader = new FileReader();
            reader.onload = (e) => addImage(e.target.result, file.name);
            reader.readAsDataURL(file);
        });
    }

    fileInput.addEventListener('change', (e) => { processFiles(e.target.files); e.target.value = ''; });
    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload)   btnUpload.addEventListener('click',  () => fileInput.click());
    if (btnRemove)   btnRemove.addEventListener('click',  () => { if (selectedIdx >= 0) removeImage(selectedIdx); });
    if (btnClear)    btnClear.addEventListener('click',   clearAll);

    dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
    });

    if (promptInput) promptInput.addEventListener('input', syncPromptToGradio);
    if (loraSelect)  loraSelect.addEventListener('change', syncLoraToGradio);

    window.__setPrompt = function(text) { if (promptInput) { promptInput.value = text; syncPromptToGradio(); } };
    window.__setLora   = function(lora) {
        if (loraSelect) {
            loraSelect.value = lora;
            loraSelect.dispatchEvent(new Event('change', {bubbles:true}));
            syncLoraToGradio();
        }
    };

    document.querySelectorAll('.example-card[data-idx]').forEach(card => {
        card.addEventListener('click', () => {
            const idx = card.getAttribute('data-idx');
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
            card.classList.add('loading');
            showToast('Loading example\u2026', 'info');
            setGradioValue('example-result-data', '');
            setGradioValue('example-idx-input', idx);
            setTimeout(() => {
                const btn = document.getElementById('example-load-btn');
                if (btn) { const b = btn.querySelector('button'); if (b) b.click(); else btn.click(); }
            }, 150);
            setTimeout(() => card.classList.remove('loading'), 12000);
        });
    });

    function syncSlider(customId, gradioId) {
        const slider  = document.getElementById(customId);
        const valSpan = document.getElementById(customId + '-val');
        if (!slider) return;
        slider.addEventListener('input', () => {
            if (valSpan) valSpan.textContent = slider.value;
            const container = document.getElementById(gradioId);
            if (!container) return;
            container.querySelectorAll('input[type="range"],input[type="number"]').forEach(el => {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, slider.value);
                    el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        });
    }
    syncSlider('custom-seed',     'gradio-seed');
    syncSlider('custom-guidance', 'gradio-guidance');
    syncSlider('custom-steps',    'gradio-steps');
    syncSlider('custom-batch',    'gradio-batch-count');

    ['custom-seed', 'custom-steps', 'custom-batch'].forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('input', saveUiState);
        el.addEventListener('change', saveUiState);
    });

    const randCheck = document.getElementById('custom-randomize');
    if (randCheck) {
        randCheck.addEventListener('change', () => {
            const container = document.getElementById('gradio-randomize');
            if (!container) return;
            const cb = container.querySelector('input[type="checkbox"]');
            if (cb && cb.checked !== randCheck.checked) cb.click();
        });
    }

    function showLoader() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.add('active');
        const sb = document.querySelector('.sb-fixed');
        if (sb) sb.textContent = 'Processing\u2026';
        if (!window.__baseDocumentTitle) window.__baseDocumentTitle = document.title || 'Qwen Image Edit';
        const total = window.__qwenBatchTotal || 1;
        const current = window.__qwenBatchCurrent || 1;
        document.title = `(${current}/${total}) - ` + window.__baseDocumentTitle;
    }
    function hideLoader() {
        if (window.__qwenBatchRunning) return;
        const l = document.getElementById('output-loader');
        if (l) l.classList.remove('active');
        const sb = document.querySelector('.sb-fixed');
        if (sb) sb.textContent = 'Done';
        if (window.__baseDocumentTitle) document.title = window.__baseDocumentTitle;
    }
    window.__showLoader = showLoader;
    window.__hideLoader = hideLoader;

    function validateBeforeRun() {
        const promptVal = promptInput ? promptInput.value.trim() : '';
        const hasImages = images.length > 0;
        if (!hasImages && !promptVal) { showToast('Please upload an image and enter a prompt', 'error'); flashPromptError(); return false; }
        if (!hasImages)  { showToast('Please upload at least one image', 'error'); return false; }
        if (!promptVal)  { showToast('Please enter an edit prompt', 'warning'); flashPromptError(); return false; }
        return true;
    }

    window.__clickGradioRunBtn = function() {
        if (!validateBeforeRun()) return;
        const batchSlider = document.getElementById('custom-batch');
        const batchValue = batchSlider ? parseInt(batchSlider.value || '1', 10) : 1;
        window.__qwenBatchRunning = batchValue > 1;
        window.__qwenBatchTotal = batchValue;
        window.__qwenBatchCurrent = 1;
        window.__lastBatchZipReadyPath = '';
        setGradioValue('gradio-batch-zip-path', '');
        setGradioValue('gradio-progress-status', `0/${batchValue}`);
        syncPromptToGradio(); syncImagesToGradio(); syncLoraToGradio(); showLoader();
        setTimeout(() => {
            const gradioBtn = document.getElementById('gradio-run-btn');
            if (!gradioBtn) return;
            const btn = gradioBtn.querySelector('button');
            if (btn) btn.click(); else gradioBtn.click();
        }, 0);
    };

    if (runBtnEl) runBtnEl.addEventListener('click', () => window.__clickGradioRunBtn());

    restoreUiState();
    renderComboWeights();
    renderGallery();
    updateCounts();
}
init();
}
"""

wire_outputs_js = r"""
() => {
function watchOutputs(attempt = 0) {
    if (window.__qwenOutputsWatchReady) return;
    const resultContainer = document.getElementById('gradio-result');
    const pathContainer   = document.getElementById('gradio-result-path');
    const batchPathContainer = document.getElementById('gradio-batch-zip-path');
    const progressContainer = document.getElementById('gradio-progress-status');
    const outBody = document.getElementById('output-image-container');
    const outPh   = document.getElementById('output-placeholder');
    const dlBtn   = document.getElementById('dl-btn-output');
    const batchDlBtn = document.getElementById('dl-btn-batch-output');
    const batchSlider = document.getElementById('custom-batch');

    if (!resultContainer || !outBody) {
        if (attempt < 40) setTimeout(() => watchOutputs(attempt + 1), 500);
        return;
    }
    window.__qwenOutputsWatchReady = true;

    function playCompletionTone() {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const ctx = new AudioCtx();
            const now = ctx.currentTime;
            const gain = ctx.createGain();
            gain.gain.setValueAtTime(0.0001, now);
            gain.gain.exponentialRampToValueAtTime(0.14, now + 0.01);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.42);
            gain.connect(ctx.destination);

            const oscA = ctx.createOscillator();
            oscA.type = 'sine';
            oscA.frequency.setValueAtTime(1318, now);
            oscA.frequency.exponentialRampToValueAtTime(1046, now + 0.18);
            oscA.connect(gain);
            oscA.start(now);
            oscA.stop(now + 0.20);

            const oscB = ctx.createOscillator();
            oscB.type = 'sine';
            oscB.frequency.setValueAtTime(1760, now + 0.015);
            oscB.frequency.exponentialRampToValueAtTime(1318, now + 0.22);
            oscB.connect(gain);
            oscB.start(now + 0.015);
            oscB.stop(now + 0.24);
        } catch (e) {
            console.warn('Completion tone failed:', e);
        }
    }

    function syncTitleProgress() {
        const progressEl = progressContainer ? (progressContainer.querySelector('textarea') || progressContainer.querySelector('input')) : null;
        const progressVal = progressEl ? progressEl.value.trim() : '';
        if (!progressVal) return;
        const parts = progressVal.split('/');
        if (parts.length === 2) {
            const current = parseInt(parts[0] || '1', 10);
            const total = parseInt(parts[1] || '1', 10);
            if (!Number.isNaN(current)) window.__qwenBatchCurrent = current;
            if (!Number.isNaN(total)) window.__qwenBatchTotal = total;
        }
        if (window.__qwenBatchRunning && window.__baseDocumentTitle) {
            document.title = `Running (${window.__qwenBatchCurrent || 1}/${window.__qwenBatchTotal || 1}) - ` + window.__baseDocumentTitle;
        }
    }

    function triggerBatchDownload(savedPath) {
        if (!savedPath) return;
        const a = document.createElement('a');
        a.href = `/gradio_api/file=${encodeURIComponent(savedPath)}`;
        a.download = savedPath.split('/').pop() || 'qwen_batch_output.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    if (dlBtn) {
        dlBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const pathEl = pathContainer ? (pathContainer.querySelector('textarea') || pathContainer.querySelector('input')) : null;
            const savedPath = pathEl ? pathEl.value.trim() : '';
            if (savedPath) {
                const a = document.createElement('a');
                a.href = `/gradio_api/file=${encodeURIComponent(savedPath)}`;
                a.download = savedPath.split('/').pop() || 'qwen_edit_output.png';
                document.body.appendChild(a); a.click(); document.body.removeChild(a);
            }
        });
    }

    if (batchDlBtn) {
        batchDlBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const pathEl = batchPathContainer ? (batchPathContainer.querySelector('textarea') || batchPathContainer.querySelector('input')) : null;
            const savedPath = pathEl ? pathEl.value.trim() : '';
            if (savedPath) {
                triggerBatchDownload(savedPath);
            }
        });
    }

    function syncDownloadButtons() {
        const pathEl = pathContainer ? (pathContainer.querySelector('textarea') || pathContainer.querySelector('input')) : null;
        const savedPath = pathEl ? pathEl.value.trim() : '';
        if (dlBtn) {
            if (savedPath) dlBtn.classList.add('visible');
            else dlBtn.classList.remove('visible');
        }

        const batchValue = batchSlider ? parseInt(batchSlider.value || '1', 10) : 1;
        const batchPathEl = batchPathContainer ? (batchPathContainer.querySelector('textarea') || batchPathContainer.querySelector('input')) : null;
        const batchSavedPath = batchPathEl ? batchPathEl.value.trim() : '';
        if (batchDlBtn) {
            if (batchValue <= 1) {
                window.__qwenBatchRunning = false;
                batchDlBtn.classList.remove('visible');
                batchDlBtn.classList.add('disabled');
                batchDlBtn.setAttribute('title', 'Save ZIP is available only when Batch is greater than 1');
            } else if (batchSavedPath) {
                window.__qwenBatchRunning = false;
                batchDlBtn.classList.add('visible');
                batchDlBtn.classList.remove('disabled');
                batchDlBtn.setAttribute('title', 'Download batch zip');
                if (window.__hideLoader) window.__hideLoader();
                if (window.__lastBatchZipReadyPath !== batchSavedPath) {
                    window.__lastBatchZipReadyPath = batchSavedPath;
                    playCompletionTone();
                    setTimeout(() => triggerBatchDownload(batchSavedPath), 120);
                }
            } else if (window.__qwenBatchRunning) {
                batchDlBtn.classList.remove('visible');
                batchDlBtn.classList.add('disabled');
                batchDlBtn.setAttribute('title', 'Batch is running');
            } else {
                batchDlBtn.classList.add('visible');
                batchDlBtn.classList.add('disabled');
                batchDlBtn.setAttribute('title', 'Batch zip will be available after batch completes');
            }
        }
    }

    function syncImage() {
        const resultImg = resultContainer.querySelector('img');
        const pathEl = pathContainer ? (pathContainer.querySelector('textarea') || pathContainer.querySelector('input')) : null;
        const savedPath = pathEl ? pathEl.value.trim() : '';

        let previewSrc = '';
        if (resultImg && resultImg.src) {
            previewSrc = resultImg.src;
        } else if (savedPath) {
            previewSrc = `/gradio_api/file=${encodeURIComponent(savedPath)}`;
        }

        if (!previewSrc) return;

        if (outPh) outPh.style.display = 'none';
        let existing = outBody.querySelector('img.modern-out-img');
        if (!existing) {
            existing = document.createElement('img');
            existing.className = 'modern-out-img';
            outBody.appendChild(existing);
        }
        if (existing.src !== previewSrc) {
            existing.src = previewSrc;
            syncDownloadButtons();
            if (window.__hideLoader) window.__hideLoader();
        }
    }
    const observer = new MutationObserver(syncImage);
    observer.observe(resultContainer, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
    if (batchPathContainer) {
        const batchObserver = new MutationObserver(syncDownloadButtons);
        batchObserver.observe(batchPathContainer, {childList:true, subtree:true, characterData:true, attributes:true});
    }
    if (pathContainer) {
        const pathObserver = new MutationObserver(() => {
            syncDownloadButtons();
            syncImage();
        });
        pathObserver.observe(pathContainer, {childList:true, subtree:true, characterData:true, attributes:true});
    }
    if (progressContainer) {
        const progressObserver = new MutationObserver(syncTitleProgress);
        progressObserver.observe(progressContainer, {childList:true, subtree:true, characterData:true, attributes:true});
    }
    if (batchSlider) {
        batchSlider.addEventListener('input', syncDownloadButtons);
        batchSlider.addEventListener('change', syncDownloadButtons);
    }
    setInterval(() => {
        syncDownloadButtons();
        syncTitleProgress();
        syncImage();
    }, 700);
    syncDownloadButtons();
    syncTitleProgress();
    syncImage();
}
watchOutputs();

function watchSeed(attempt = 0) {
    if (window.__qwenSeedWatchReady) return;
    const seedContainer = document.getElementById('gradio-seed');
    const seedSlider    = document.getElementById('custom-seed');
    const seedVal       = document.getElementById('custom-seed-val');
    if (!seedContainer || !seedSlider) {
        if (attempt < 40) setTimeout(() => watchSeed(attempt + 1), 500);
        return;
    }
    window.__qwenSeedWatchReady = true;
    function sync() {
        const el = seedContainer.querySelector('input[type="range"],input[type="number"]');
        if (el && el.value) { seedSlider.value = el.value; if (seedVal) seedVal.textContent = el.value; }
    }
    const obs = new MutationObserver(sync);
    obs.observe(seedContainer, {childList:true, subtree:true, attributes:true, attributeFilter:['value']});
    sync();
}
watchSeed();

function watchExampleResults(attempt = 0) {
    if (window.__qwenExampleWatchReady) return;
    const container = document.getElementById('example-result-data');
    if (!container) {
        if (attempt < 40) setTimeout(() => watchExampleResults(attempt + 1), 500);
        return;
    }
    window.__qwenExampleWatchReady = true;
    let lastProcessed = '';

    function checkResult() {
        const el  = container.querySelector('textarea') || container.querySelector('input');
        if (!el) return;
        const val = el.value;
        if (!val || val === lastProcessed || val.length < 20) return;
        try {
            const data = JSON.parse(val);
            if (data.status === 'ok' && data.images && data.images.length > 0) {
                lastProcessed = val;
                if (window.__setPrompt && data.prompt) window.__setPrompt(data.prompt);
                if (window.__setLora   && data.lora)   window.__setLora(data.lora);
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                if (window.__showToast) window.__showToast('Example prompt/model loaded', 'info');
            } else if (data.status === 'error') {
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                if (window.__showToast) window.__showToast('Could not load example images', 'error');
            }
        } catch(e) { console.error('Example parse error:', e); }
    }

    const obs = new MutationObserver(checkResult);
    obs.observe(container, {childList:true, subtree:true, characterData:true, attributes:true});
    checkResult();
}
watchExampleResults();
}
"""

ROCKET_LOGO_SVG = '''<svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2C12 2 7 6.5 7 13c0 1.4.3 2.7.8 3.9L5 19.7l1.4 1.4 2.8-2.8c1.2.5 2.5.7 3.8.7s2.6-.2 3.8-.7l2.8 2.8 1.4-1.4-2.8-2.8c.5-1.2.8-2.5.8-3.9 0-6.5-5-11-5-11z"/>
  <circle cx="12" cy="13" r="2"/>
  <path d="M9 21c0 1.1.9 2 2 2h2c1.1 0 2-.9 2-2v-1H9v1z"/>
</svg>'''

UPLOAD_SVG   = '<svg class="tb-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
REMOVE_SVG   = '<svg class="tb-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
CLEAR_SVG    = '<svg class="tb-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>'
DOWNLOAD_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-5-5h3V4h4v7h3l-5 5z" fill="currentColor"/><path d="M20 18H4v2h16v-2z" fill="currentColor"/></svg>'
GITHUB_SVG   = '<svg width="15" height="15" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path fill="#ffffff" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'

LORA_OPTIONS_HTML = "\n".join(
    f'<option value="{html_lib.escape(name)}">{html_lib.escape(name)}</option>'
    for name in ADAPTER_NAMES
)

with gr.Blocks() as demo:

    hidden_images_b64 = gr.Textbox(value="[]",  elem_id="hidden-images-b64",   elem_classes="hidden-input", container=False)
    prompt            = gr.Textbox(value="",    elem_id="prompt-gradio-input",  elem_classes="hidden-input", container=False)
    lora_adapter      = gr.Dropdown(choices=ADAPTER_NAMES, value="XY", elem_id="gradio-lora", elem_classes="hidden-input", container=False)
    resolution_preset = gr.Textbox(value="Auto", elem_id="gradio-resolution-preset", elem_classes="hidden-input", container=False)
    combo_weights     = gr.Textbox(value="", elem_id="gradio-combo-weights", elem_classes="hidden-input", container=False)
    seed              = gr.Slider(minimum=0, maximum=MAX_SEED, step=1, value=0, elem_id="gradio-seed",       elem_classes="hidden-input", container=False)
    randomize_seed    = gr.Checkbox(value=True, elem_id="gradio-randomize",     elem_classes="hidden-input", container=False)
    guidance_scale    = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.0, elem_id="gradio-guidance", elem_classes="hidden-input", container=False)
    steps             = gr.Slider(minimum=1, maximum=50, step=1, value=4,       elem_id="gradio-steps",      elem_classes="hidden-input", container=False)
    batch_count       = gr.Slider(minimum=1, maximum=100, step=1, value=1,      elem_id="gradio-batch-count", elem_classes="hidden-input", container=False)
    result            = gr.Image(elem_id="gradio-result",                       elem_classes="hidden-input", container=False, format="png")
    result_path       = gr.Textbox(value="", elem_id="gradio-result-path",      elem_classes="hidden-input", container=False)
    batch_zip_path    = gr.Textbox(value="", elem_id="gradio-batch-zip-path",   elem_classes="hidden-input", container=False)
    progress_status   = gr.Textbox(value="", elem_id="gradio-progress-status",  elem_classes="hidden-input", container=False)

    example_idx      = gr.Textbox(value="", elem_id="example-idx-input",   elem_classes="hidden-input", container=False)
    example_result   = gr.Textbox(value="", elem_id="example-result-data",  elem_classes="hidden-input", container=False)
    example_load_btn = gr.Button("Load Example", elem_id="example-load-btn")

    gr.HTML(f"""
    <div class="app-shell">

      <!-- Toolbar -->
      <div class="app-toolbar">
        <button id="tb-upload" class="modern-tb-btn" title="Upload images">
          {UPLOAD_SVG}<span class="tb-label">Upload</span>
        </button>
        <button id="tb-remove" class="modern-tb-btn" title="Remove selected image">
          {REMOVE_SVG}<span class="tb-label">Remove</span>
        </button>
        <button id="tb-clear" class="modern-tb-btn" title="Clear all images">
          {CLEAR_SVG}<span class="tb-label">Clear All</span>
        </button>
        <div class="tb-sep"></div>
        <span id="tb-image-count" class="tb-info">No images</span>
      </div>

      <!-- Main row -->
      <div class="app-main-row">

        <!-- Left -->
        <div class="app-main-left">

          <div id="gallery-drop-zone">
            <div id="upload-prompt" class="upload-prompt-modern">
              <div id="upload-click-area" class="upload-click-area">
                <svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="8" y="14" width="64" height="52" rx="6" fill="none"
                        stroke="#FF4500" stroke-width="2" stroke-dasharray="4 3"/>
                  <polygon points="12,62 30,40 42,50 54,34 68,62"
                           fill="rgba(255,69,0,0.12)" stroke="#FF4500" stroke-width="1.5"/>
                  <circle cx="28" cy="30" r="6"
                          fill="rgba(255,69,0,0.18)" stroke="#FF4500" stroke-width="1.5"/>
                </svg>
                <span class="upload-main-text">Click or drag images here</span>
              </div>
            </div>
            <input id="custom-file-input" type="file" accept="image/*" multiple style="display:none;" />
            <div id="image-gallery-grid" class="image-gallery-grid" style="display:none;"></div>
          </div>

          <div class="panel-card">
            <div class="panel-card-title">Edit Instruction</div>
            <div class="panel-card-body">
              <label class="modern-label" for="custom-prompt-input">Prompt</label>
              <textarea id="custom-prompt-input" class="modern-textarea" rows="6"
                        placeholder="e.g., transform into anime, upscale, change lighting&hellip;"></textarea>
            </div>
          </div>

          <div class="settings-group">
            <div class="settings-group-title">Advanced Settings</div>
            <div class="settings-group-body">
              <div id="combo-weights-panel" class="combo-weights-panel">
                <div class="combo-weights-title">Combo LoRA Weights</div>
                <div id="combo-weights-body"></div>
              </div>
              <label class="modern-label" for="custom-resolution-preset" style="margin-bottom:8px;">Resolution</label>
              <select id="custom-resolution-preset" class="lora-native-select" style="margin-bottom:14px;">
                <option value="Auto" selected>Auto</option>
                <option value="1080p Portrait">1080p Portrait</option>
                <option value="1080p Landscape">1080p Landscape</option>
                <option value="Max Portrait (1152x2048)">Max Portrait (1152x2048)</option>
                <option value="Max Landscape (2048x1152)">Max Landscape (2048x1152)</option>
                <option value="iPhone 15 Pro Portrait">iPhone 15 Pro Portrait</option>
              </select>
              <div class="slider-row">
                <label>Seed</label>
                <input type="range" id="custom-seed" min="0" max="2147483647" step="1" value="0">
                <span class="slider-val" id="custom-seed-val">0</span>
              </div>
              <div class="checkbox-row">
                <input type="checkbox" id="custom-randomize" checked>
                <label for="custom-randomize">Randomize seed</label>
              </div>
              <div class="slider-row">
                <label>Guidance</label>
                <input type="range" id="custom-guidance" min="1" max="10" step="0.1" value="1.0">
                <span class="slider-val" id="custom-guidance-val">1.0</span>
              </div>
              <div class="slider-row">
                <label>Steps</label>
                <input type="range" id="custom-steps" min="1" max="50" step="1" value="4">
                <span class="slider-val" id="custom-steps-val">4</span>
              </div>
              <div class="slider-row">
                <label>Batch</label>
                <input type="range" id="custom-batch" min="1" max="100" step="1" value="1">
                <span class="slider-val" id="custom-batch-val">1</span>
              </div>
            </div>
          </div>

        </div><!-- /left -->

        <!-- Right -->
        <div class="app-main-right">

          <div class="lora-selector-card">
            <div class="lora-selector-body">
              <div class="lora-select-label">Editing Style / LoRA</div>
              <select id="custom-lora-select" class="lora-native-select">
                {LORA_OPTIONS_HTML}
              </select>
            </div>
          </div>

          <div style="padding:14px 18px 6px;">
            <button id="custom-run-btn" class="btn-run">
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C12 2 7 6.5 7 13c0 1.4.3 2.7.8 3.9L5 19.7l1.4 1.4 2.8-2.8c1.2.5 2.5.7 3.8.7s2.6-.2 3.8-.7l2.8 2.8 1.4-1.4-2.8-2.8c.5-1.2.8-2.5.8-3.9 0-6.5-5-11-5-11z"/>
                <circle cx="12" cy="13" r="2"/>
              </svg>
              <span id="run-btn-label">Edit Image</span>
            </button>
          </div>

          <div class="output-frame" style="flex:1">
            <div class="out-title">
              <span>Output</span>
              <span id="dl-btn-batch-output" class="out-download-btn disabled" title="Batch zip will be available after batch completes">
                {DOWNLOAD_SVG} Save ZIP
              </span>
              <span id="dl-btn-output" class="out-download-btn" title="Download result">
                {DOWNLOAD_SVG} Save
              </span>
            </div>
            <div class="out-body" id="output-image-container">
              <div class="modern-loader" id="output-loader">
                <div class="loader-spinner"></div>
                <div class="loader-text">Processing image&hellip;</div>
                <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
              </div>
              <div class="out-placeholder" id="output-placeholder">Result will appear here</div>
            </div>
          </div>

        </div><!-- /right -->
      </div><!-- /main-row -->

      <div class="examples-section">
        <div class="examples-title">Quick Examples &mdash; click to load</div>
        <div class="examples-scroll">
          {EXAMPLE_CARDS_HTML}
        </div>
      </div>

      <div class="app-statusbar">
        <div class="sb-section" id="sb-image-count">No images uploaded</div>
        <div class="sb-section sb-fixed">Ready</div>
      </div>

    </div><!-- /app-shell -->
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=None, js=gallery_js)
    demo.load(fn=None, js=wire_outputs_js)

    run_btn.click(
        fn=infer,
        inputs=[hidden_images_b64, prompt, lora_adapter, resolution_preset, combo_weights, seed, randomize_seed, guidance_scale, steps, batch_count],
        outputs=[result, seed, result_path, batch_zip_path, progress_status],
        js=r"""(imgs, p, la, rp, cw, s, rs, gs, st, bc) => {
            const images    = window.__uploadedImages || [];
            const b64Array  = images.map(img => img.b64);
            const imgsJson  = JSON.stringify(b64Array);
            const promptEl  = document.getElementById('custom-prompt-input');
            const loraEl    = document.getElementById('custom-lora-select');
            const presetEl  = document.getElementById('custom-resolution-preset');
            const batchEl   = document.getElementById('custom-batch');
            const comboWeightEls = Array.from(document.querySelectorAll('.combo-weight-slider'));
            const promptVal = promptEl ? promptEl.value : p;
            const loraVal   = loraEl   ? loraEl.value   : la;
            const presetVal = presetEl ? presetEl.value : rp;
            const comboWeightsVal = comboWeightEls.length ? JSON.stringify(comboWeightEls.map(el => parseFloat(el.value || '1'))) : '';
            const batchVal  = batchEl ? parseInt(batchEl.value || '1', 10) : bc;
            return [imgsJson, promptVal, loraVal, presetVal, comboWeightsVal, s, rs, gs, st, batchVal];
        }""",
    )

    example_load_btn.click(
        fn=load_example_data,
        inputs=[example_idx],
        outputs=[example_result],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(
        css=css,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=["examples", OUTPUT_DIR],
        share=True,
        inline=False,
        debug=True,
        prevent_thread_lock=True,
        server_name="0.0.0.0",
        server_port=None,
    )
