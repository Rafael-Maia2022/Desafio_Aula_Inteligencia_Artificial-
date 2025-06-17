from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

prompt = "car"

width = 1024

height = 1024

image = pipeline(prompt,width=width, height=height).images[0]

image.save("generative_image_car.png")

image.show()