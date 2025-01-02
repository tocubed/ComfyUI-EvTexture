import torch
from dataclasses import dataclass


@dataclass
class EventSimulatorConfig:
    contrast_threshold_pos: float = 0.275
    contrast_threshold_neg: float = 0.275
    refractory_period: float = 1e-4


def events(
    img: torch.Tensor,
    time: float,
    last_img: torch.Tensor,
    last_time: float,
    last_event_timestamp: torch.Tensor,
    ref_values: torch.Tensor,
    config: EventSimulatorConfig,
):
    delta = img - last_img
    delta_t = time - last_time

    pol = torch.where(delta >= 0.0, 1.0, -1.0)
    contrast_threshold = torch.where(
        pol > 0, config.contrast_threshold_pos, config.contrast_threshold_neg
    )

    active_mask = delta.abs() > 1e-6
    events = []
    curr_cross = ref_values.clone()
    while True:
        curr_cross[active_mask] += pol[active_mask] * contrast_threshold[active_mask]
        pos_crossing = (pol > 0) & (curr_cross > last_img) & (curr_cross <= img)
        neg_crossing = (pol < 0) & (curr_cross < last_img) & (curr_cross >= img)
        crossing_conditions = pos_crossing | neg_crossing

        active_mask &= crossing_conditions

        if not active_mask.any():  # loop until no activations
            break

        ref_values[active_mask] = curr_cross[active_mask]

        edt = torch.zeros_like(img)
        edt[active_mask] = (
            (curr_cross[active_mask] - last_img[active_mask])
            * delta_t
            / delta[active_mask]
        )
        t = last_time + edt

        event_mask = (t - last_event_timestamp) >= config.refractory_period
        event_mask |= last_event_timestamp == 0
        event_mask &= active_mask
        last_event_timestamp[event_mask] = t[event_mask]

        indices = event_mask.argwhere()
        ys, xs = indices[:, 0], indices[:, 1]
        events.append(torch.column_stack((xs, ys, t[ys, xs], pol[ys, xs])))

    if events:
        events = torch.vstack(events)
    else:
        events = torch.empty((0, 4))
    events = events[events[:, 2].argsort()]
    return events


def events_generator(imgs, timestamps, config: EventSimulatorConfig):
    it = iter(zip(imgs, timestamps))
    last_img, last_time = next(it)
    last_img = last_img.squeeze()

    assert len(last_img.shape) == 2, "expected single channel images of shape [h, w]"

    last_event_timestamp = torch.zeros_like(last_img)
    ref_values = last_img.clone()

    for img, time in it:
        img = img.squeeze()
        yield events(
            img, time, last_img, last_time, last_event_timestamp, ref_values, config
        )
        last_img = img
        last_time = time


def events_to_image(events, height: int, width: int):
    img = torch.zeros((3, height, width), dtype=torch.float32)
    xs = events[:, 0].long()
    ys = events[:, 1].long()
    pol = events[:, 3]
    pos_mask = pol > 0
    neg_mask = pol < 0
    img[0, ys[pos_mask], xs[pos_mask]] = 1.0
    img[2, ys[neg_mask], xs[neg_mask]] = 1.0
    return img
