import torch
import os
from diffusers import DiffusionPipeline, StableDiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, DPMSolverMultistepScheduler
# from transformers import AutoTokenizer
import questionary

# settings:
prompt = "A person holding a bat while jumping on the trampoline inside a giant AI data center"
negativePrompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,"
numPhotos = 4
inferenceSteps = 28
height = 768
width = 768
guidanceScale = 7.5
maxSequenceLength = 32
useCpu = False
useLocalModels = True
useQuantization = False


model = questionary.select(
    "Which model to use:",
    choices=[
        "FLUX.1 Schnell",
        "StableDiffusion_3.5_medium",
        "StableDiffusion_2.1",
        "StableDiffusion_1.4"
    ]
).ask()

localModelList = {
    "FLUX.1 Schnell": "/media/philip/Games/Users/phili/Documents/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9",
    "StableDiffusion_3.5_medium": "/media/philip/Games/Users/phili/Documents/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium",
    "StableDiffusion_2.1": "/media/philip/Games/Users/phili/Documents/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1",
    "StableDiffusion_1.4": "/media/philip/Games/Users/phili/Documents/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
}

modelList = {
    "FLUX.1 Schnell": "black-forest-labs/FLUX.1-schnell",
    "StableDiffusion_3.5_medium": "stabilityai/stable-diffusion-3.5-medium",
    "StableDiffusion_2.1": "stabilityai/stable-diffusion-2-1",
    "StableDiffusion_1.4": "https://huggingface.co/CompVis/stable-diffusion-v1-4"
}

currentModel = localModelList.get(model, modelList[model])


def makeFluxPipeline(model):
    return FluxPipeline.from_pretrained(
        currentModel,
        torch_dtype=torch.float16,
        # device_map="balanced",
        quantization_config=quantConfig
    ).to('cuda')

def makeStableDiffusionPipeline(model):
    return StableDiffusionPipeline.from_pretrained(
        currentModel,
        torch_dtype=torch.float16,
        quantization_config = quantConfig
    ).to('cuda')

def makeStableDiffusion3Pipeline(model):
    return StableDiffusion3Pipeline.from_pretrained(
        currentModel,
        torch_dtype=torch.float16,
        quantization_config = quantConfig
    ).to('cuda')

def generateFluxResponse(prompt):
    return pipe(
        prompt,
        guidance_scale = guidanceScale,
        num_inference_steps = inferenceSteps,
        max_sequence_length = maxSequenceLength,
        height = height,
        width = width,
        num_images_per_prompt=numPhotos,
        generator = torch.Generator("cuda").manual_seed(0)
    )

def generateStableDiffusionResponse(prompt):
    return pipe(
        prompt,
        negative_prompt=negativePrompt,
        guidance_scale=guidanceScale,
        num_inference_steps=inferenceSteps,
        height=height,
        width=width,
        num_images_per_prompt=numPhotos,
        generator=torch.Generator("cuda").manual_seed(0)
    )

def setPipeExtraSettings(useCpu):
    if useCpu:
        pipe.enable_model_cpu_offload()
    # pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

def quantizeModel():
    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4"
    )

if useQuantization:
    quantConfig = quantizeModel()
else:
    quantConfig = None


image = []

if model == "FLUX.1 Schnell": # flux pipeline
    pipe = makeFluxPipeline(model)
    setPipeExtraSettings(useCpu)
    for img in generateFluxResponse(prompt).images:
        image.append(img)
elif model == "StableDiffusion_1.4" or model == "StableDiffusion_2.1": # stable diffusion pipeline
    pipe = makeStableDiffusionPipeline(model)
    setPipeExtraSettings(useCpu)
    if model == "StableDiffusion_2.1":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    for img in generateStableDiffusionResponse(prompt).images:
        image.append(img)
elif model == "StableDiffusion_3.5_medium": # stable diffusion 3 pipeline
    pipe = makeStableDiffusion3Pipeline(model)
    setPipeExtraSettings(useCpu)
    for img in generateStableDiffusionResponse(prompt).images:
        image.append(img)


script_dir = os.path.dirname(os.path.abspath(__file__))

for idx, img in enumerate(image):
    filename = os.path.join(script_dir, "images", f"image_{idx}_{model}.png")
    img.save(filename)