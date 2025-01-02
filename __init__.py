import torch
from comfy import model_management
import folder_paths
from .evtexture.evtexture_arch import EvTexture
from .esim import events_generator, events_to_image, EventSimulatorConfig
from .evoxels import package_bidirectional_event_voxels

EVENTS_TYPE = "EVT_EVENTS"
EVTEXTURE_MODEL_TYPE = "EVTEXTURE_MODEL"


class VideoToEvents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE", {}), "fps": ("FLOAT", {})},
        }

    RETURN_TYPES = (EVENTS_TYPE,)
    RETURN_NAMES = ("events",)
    CATEGORY = "EVTexture"
    FUNCTION = "events"

    def events(self, images, fps: float):
        imgs = torch.mean(images, dim=3)
        log_imgs = (imgs + 1e-3).log()

        timestamps = [i / fps for i in range(len(images))]
        config = EventSimulatorConfig()
        return (list(events_generator(log_imgs, timestamps, config)),)


class EventsToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "events": (EVENTS_TYPE, {"forceInput": True}),
                "width": ("INT", {}),
                "height": ("INT", {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "EVTexture"
    FUNCTION = "to_image"

    def to_image(self, events, height: int, width: int):
        b, h, w = len(events), height, width
        res = torch.zeros((b, h, w, 3))
        for i in range(b):
            res[i, :, :, :] = events_to_image(events[i], h, w).permute(1, 2, 0)
        return (res,)


class LoadEvTextureModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
            }
        }

    RETURN_TYPES = (EVTEXTURE_MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    CATEGORY = "EVTexture"
    FUNCTION = "load"

    def load(self, model_name):
        path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        model = EvTexture()
        params_dict = torch.load(path, weights_only=True)["params_ema"]
        model.load_state_dict(params_dict, strict=True)
        return (model,)


class EvTextureUpscaleVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "events": (EVENTS_TYPE, {}),
                "model": (EVTEXTURE_MODEL_TYPE, {}),
                "fps": ("FLOAT", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "EVTexture"
    FUNCTION = "upscale"

    def upscale(self, images, events, model: EvTexture, fps: float):
        device = model_management.get_torch_device()

        n, h, w, _ = images.shape
        memory_required = model_management.module_size(model)
        memory_required += (
            n * (h * w * 3) * images.element_size() * 4 * 384.0
        )  # The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += images.nelement() * images.element_size()
        model_management.free_memory(memory_required, device)

        model.to(device)
        imgs = images.movedim(-1, -3).unsqueeze(0).to(device)
        events = torch.vstack(events).to(device)

        xs, ys, ts, pols = events.T
        timestamps = [i / fps for i in range(n)]
        bins = 5
        voxels_f = torch.stack(
            package_bidirectional_event_voxels(
                xs, ys, ts, pols, timestamps, False, bins, (h, w)
            )
        ).unsqueeze(0)
        voxels_b = torch.stack(
            package_bidirectional_event_voxels(
                xs, ys, ts, pols, timestamps, True, bins, (h, w)
            )
        ).unsqueeze(0)
        del xs, ys, ts, pols, events

        s = model.forward(imgs, voxels_f, voxels_b)[0].to("cpu")

        model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)


NODE_CLASS_MAPPINGS = {
    "EVTVideoToEvents": VideoToEvents,
    "EVTEventsToImage": EventsToImage,
    "EVTLoadEvTextureModel": LoadEvTextureModel,
    "EVTTextureUpscaleVideo": EvTextureUpscaleVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EVTVideoToEvents": "Video to Camera Events",
    "EVTEventsToImage": "Camera Events To Images",
    "EVTLoadEvTextureModel": "Load EvTexture Model",
    "EVTTextureUpscaleVideo": "EvTexture Video Upscale",
}
