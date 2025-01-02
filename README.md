# ComfyUI wrapper nodes for [EvTexture](https://github.com/DachunKai/EvTexture)

Get pre-trained models as described [here](https://github.com/DachunKai/EvTexture?tab=readme-ov-file#test) and place them in upscale_models folder.
At the time the pre-trained models works better on low resolution images (64-256 px) packed with detail. The Vimeo90K version seems to work best (less artefacts) for larger images. 

Example Workflow:

![workflow](https://github.com/user-attachments/assets/f1ed1670-d8b7-4707-a4a7-80d8ef1e7c95)

You can also combine with RIFE or other interpolation model to generate events at a higher FPS before feeding them into the upscaler.
