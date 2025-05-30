import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "/media/philip/Games/Users/phili/Documents/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b",
    torch_dtype=torch.float16
).to("cuda")

prompt = "A cat holding a sign that says hello world"

for i in range(10):
    image = pipe(
        prompt,
        num_inference_steps=10 * (i * i + 1)
    ).images[0]
    image.save("images/image_" + str(i) + "stableDiffusion.png")




