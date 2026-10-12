"""Open-set objects on the two recordings, then which of them do not match.

Grounding DINO finds bottles, curtains, and the rest from a text list.
SAM 2 tightens each box to the object.
DINOv2 says whether a crop in the second visit is the same thing.
SigLIP checks the written label against the crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from spatial_change.changes import Change, ChangeResult, resync_summary
from spatial_change.viz import _draw_tag

LABELS = (
    "bottle",
    "soap dispenser",
    "curtain",
    "towel",
    "cup",
    "jug",
    "toothbrush",
    "bag",
    "box",
    "toilet",
    "sink",
    "faucet",
    "drain",
    "shower head",
)
PROMPT = ". ".join(LABELS) + "."
GROUNDING_ID = "IDEA-Research/grounding-dino-tiny"
SAM_IDS = ("facebook/sam2.1-hiera-tiny", "facebook/sam2-hiera-tiny")
DINO_ID = "facebook/dinov2-small"
SIGLIP_ID = "google/siglip-base-patch16-224"

_MODELS: dict | None = None


@dataclass
class _Det:
    label: str
    score: float
    box: np.ndarray
    frame_index: int
    embedding: np.ndarray


def compare_visits(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
) -> tuple[ChangeResult, np.ndarray, np.ndarray] | None:
    """Return the change sentences and the two frames with boxes on the real objects."""
    if len(frames_a) < 1 or len(frames_b) < 1:
        return None
    models = _load()
    print("  detecting objects in the first recording")
    before = _represent(frames_a, models)
    print("  detecting objects in the second recording")
    after = _represent(frames_b, models)
    pairs, only_before, only_after = _match(before, after)
    only_before = sorted(only_before, key=lambda det: -det.score)[:6]
    only_after = sorted(only_after, key=lambda det: -det.score)[:6]
    changes: list[Change] = []
    for det in only_before:
        det.box = _tighten(frames_a[det.frame_index], det.box, models)
        changes.append(_make("removed", det, frames_a[det.frame_index].shape))
    for det in only_after:
        det.box = _tighten(frames_b[det.frame_index], det.box, models)
        changes.append(_make("added", det, frames_b[det.frame_index].shape))
    changes.sort(key=lambda change: (-change.size, change.kind))
    result = ChangeResult(changes=changes)
    resync_summary(result)
    if not changes:
        result.summary = (
            "The open-set detector looked for bottles, curtains, and the other listed objects. "
            "Nothing confident was in only one of the two recordings."
        )
    marked_before = _paint(frames_a, only_before, "REMOVED")
    marked_after = _paint(frames_b, only_after, "ADDED")
    print(
        f"  open-set: {len(pairs)} objects in both visits, "
        f"{len(only_before)} only in the first, {len(only_after)} only in the second"
    )
    return result, marked_before, marked_after


def _pretrained_pair(processor_cls, model_cls, model_ids: tuple[str, ...], device: torch.device):
    error = None
    for model_id in model_ids:
        try:
            processor = processor_cls.from_pretrained(model_id)
            model = model_cls.from_pretrained(model_id).to(device).eval()
            return processor, model
        except Exception as exc:
            error = exc
            print(f"  could not load {model_id}: {exc}")
    raise RuntimeError(f"Could not load SAM 2 ({error})")


def _load() -> dict:
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    print("  loading Grounding DINO, SAM 2, DINOv2, and SigLIP")
    device = torch.device("cpu")
    grounding_processor = AutoProcessor.from_pretrained(GROUNDING_ID)
    grounding = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_ID).to(device).eval()
    try:
        sam_processor, sam = _pretrained_pair(Sam2Processor, Sam2Model, SAM_IDS, device)
    except Exception as exc:
        print(f"  SAM 2 stayed on the Grounding DINO box ({exc})")
        sam_processor, sam = None, None
    dino_processor = AutoImageProcessor.from_pretrained(DINO_ID)
    dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()
    try:
        siglip_processor = AutoProcessor.from_pretrained(SIGLIP_ID)
        siglip = AutoModel.from_pretrained(SIGLIP_ID).to(device).eval()
    except Exception as exc:
        print(f"  SigLIP label check skipped ({exc})")
        siglip_processor, siglip = None, None
    _MODELS = {
        "device": device,
        "grounding_processor": grounding_processor,
        "grounding": grounding,
        "sam_processor": sam_processor,
        "sam": sam,
        "dino_processor": dino_processor,
        "dino": dino,
        "siglip_processor": siglip_processor,
        "siglip": siglip,
    }
    return _MODELS


def _represent(frames: list[np.ndarray], models: dict) -> list[_Det]:
    found: list[_Det] = []
    for index, frame in _spread(frames, 4):
        found.extend(_detect(frame, index, models))
    groups: list[_Det] = []
    for det in sorted(found, key=lambda item: -item.score):
        host = None
        best = 0.62
        for group in groups:
            if not _same_name(group.label, det.label):
                continue
            sim = float(np.dot(group.embedding, det.embedding))
            if sim > best:
                host = group
                best = sim
        if host is None:
            groups.append(det)
        elif det.score > host.score:
            host.score = det.score
            host.box = det.box
            host.frame_index = det.frame_index
            host.embedding = det.embedding
            host.label = det.label
    return groups


def _detect(frame: np.ndarray, frame_index: int, models: dict) -> list[_Det]:
    image = Image.fromarray(frame)
    processor = models["grounding_processor"]
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(models["device"])
    with torch.inference_mode():
        outputs = models["grounding"](**inputs)
    height, width = frame.shape[:2]
    processed = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.28,
        text_threshold=0.25,
        target_sizes=[(height, width)],
    )[0]
    boxes = processed["boxes"].detach().cpu().numpy()
    scores = processed["scores"].detach().cpu().numpy()
    labels = list(processed["text_labels"])
    area = float(height * width)
    kept: list[_Det] = []
    order = np.argsort(-scores)
    for index in order:
        box = boxes[int(index)].astype(np.float64)
        box[0] = np.clip(box[0], 0, width - 1)
        box[1] = np.clip(box[1], 0, height - 1)
        box[2] = np.clip(box[2], 0, width - 1)
        box[3] = np.clip(box[3], 0, height - 1)
        box_area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
        if box_area < 0.004 * area or box_area > 0.45 * area:
            continue
        if any(_iou(box, other.box) > 0.55 for other in kept):
            continue
        crop = _crop(frame, box)
        if crop is None:
            continue
        label = _clean_label(str(labels[int(index)]))
        checked = _siglip_label(crop, models)
        if checked is not None and checked[1] >= 0.45:
            label = checked[0]
        kept.append(_Det(
            label=label,
            score=float(scores[int(index)]),
            box=box,
            frame_index=frame_index,
            embedding=_embed(crop, models),
        ))
        if len(kept) == 8:
            break
    return kept


def _match(before: list[_Det], after: list[_Det]) -> tuple[list[tuple[_Det, _Det]], list[_Det], list[_Det]]:
    unused = set(range(len(after)))
    pairs = []
    only_before = []
    for det in before:
        best_index = None
        best = 0.50
        for index in unused:
            other = after[index]
            sim = float(np.dot(det.embedding, other.embedding))
            if _same_name(det.label, other.label):
                sim += 0.08
            if sim > best:
                best = sim
                best_index = index
        if best_index is None:
            only_before.append(det)
        else:
            unused.remove(best_index)
            pairs.append((det, after[best_index]))
    only_after = [after[index] for index in sorted(unused)]
    return pairs, only_before, only_after


def _tighten(frame: np.ndarray, box: np.ndarray, models: dict) -> np.ndarray:
    if models.get("sam") is None:
        return box
    try:
        image = Image.fromarray(frame)
        processor = models["sam_processor"]
        inputs = processor(
            images=image,
            input_boxes=[[box.tolist()]],
            return_tensors="pt",
        ).to(models["device"])
        with torch.inference_mode():
            outputs = models["sam"](**inputs)
        pred = outputs.pred_masks
        if pred.ndim == 5:
            pred = pred[:, 0]
        masks = processor.post_process_masks(pred, inputs["original_sizes"])[0]
        mask = masks[0]
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask)
        if mask.ndim == 3:
            mask = mask[0]
        ys, xs = np.nonzero(mask > 0)
        if len(xs) < 30:
            return box
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    except Exception:
        return box


def _embed(crop: np.ndarray, models: dict) -> np.ndarray:
    inputs = models["dino_processor"](images=Image.fromarray(crop), return_tensors="pt").to(models["device"])
    with torch.inference_mode():
        hidden = models["dino"](**inputs).last_hidden_state[:, 0]
    vector = hidden[0].detach().cpu().numpy().astype(np.float64)
    return vector / (np.linalg.norm(vector) + 1e-8)


def _siglip_label(crop: np.ndarray, models: dict) -> tuple[str, float] | None:
    if models.get("siglip") is None:
        return None
    processor = models["siglip_processor"]
    texts = [f"a photo of a {label}" for label in LABELS]
    inputs = processor(text=texts, images=Image.fromarray(crop), padding="max_length", return_tensors="pt")
    inputs = inputs.to(models["device"])
    with torch.inference_mode():
        logits = models["siglip"](**inputs).logits_per_image[0]
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    choice = int(np.argmax(scores))
    score = float(scores[choice])
    if score < 0.12:
        return None
    return LABELS[choice], score


def _make(kind: str, det: _Det, shape: tuple) -> Change:
    height, width = shape[:2]
    center = ((det.box[0] + det.box[2]) / 2.0) / max(width, 1)
    if center < 0.40:
        place = "on the left side"
    elif center > 0.60:
        place = "on the right side"
    else:
        place = "in the middle"
    if kind == "removed":
        description = (
            f"A {det.label} {place} of the first recording is not in the second recording."
        )
    else:
        description = (
            f"A {det.label} appeared {place} of the second recording. It was not in the first."
        )
    diagonal = float(np.linalg.norm(det.box[2:] - det.box[:2]))
    return Change(
        kind=kind,
        color_name=det.label.split()[-1],
        label=det.label,
        description=description,
        centroid=np.array([center, 0.5, 0.0]),
        centroid_to=None,
        points=np.zeros((0, 3)),
        points_to=None,
        size=diagonal,
        rgb=np.array([0.85, 0.25, 0.2] if kind == "removed" else [0.2, 0.75, 0.3]),
    )


def _paint(frames: list[np.ndarray], dets: list[_Det], tag: str) -> np.ndarray:
    if not dets:
        frame = frames[len(frames) // 2]
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    counts: dict[int, int] = {}
    for det in dets:
        counts[det.frame_index] = counts.get(det.frame_index, 0) + 1
    frame_index = max(counts, key=counts.get)
    canvas = cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2BGR)
    color = (48, 64, 230) if tag == "REMOVED" else (60, 190, 70)
    for det in dets:
        if det.frame_index != frame_index:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in det.box]
        _draw_tag(canvas, (x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)), f"{tag}  {det.label.upper()}", color)
    return canvas


def _spread(frames: list[np.ndarray], count: int) -> list[tuple[int, np.ndarray]]:
    if len(frames) <= count:
        return list(enumerate(frames))
    indexes = np.linspace(0, len(frames) - 1, count).astype(int)
    return [(int(index), frames[int(index)]) for index in indexes]


def _crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def _clean_label(text: str) -> str:
    low = text.lower().strip()
    hits = [label for label in LABELS if label in low]
    if hits:
        return max(hits, key=len)
    return low or "object"


def _same_name(left: str, right: str) -> bool:
    return left == right or left in right or right in left


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(left[2] - left[0], 0) * max(left[3] - left[1], 0)
    area += max(right[2] - right[0], 0) * max(right[3] - right[1], 0)
    return float(inter / (area - inter + 1e-6))
