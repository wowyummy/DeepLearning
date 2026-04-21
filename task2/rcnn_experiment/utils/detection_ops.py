from __future__ import annotations

import torch


def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    proposal_widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    proposal_heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    proposal_ctr_x = proposals[:, 0] + 0.5 * proposal_widths
    proposal_ctr_y = proposals[:, 1] + 0.5 * proposal_heights

    gt_widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    gt_heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)
    gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - proposal_ctr_x) / proposal_widths
    dy = (gt_ctr_y - proposal_ctr_y) / proposal_heights
    dw = torch.log(gt_widths / proposal_widths)
    dh = torch.log(gt_heights / proposal_heights)
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(rel_codes: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = rel_codes[:, 0]
    dy = rel_codes[:, 1]
    dw = rel_codes[:, 2].clamp(max=4.0)
    dh = rel_codes[:, 3].clamp(max=4.0)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=1)


def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(min=0, max=max(width - 1, 0))
    clipped[:, 1::2] = clipped[:, 1::2].clamp(min=0, max=max(height - 1, 0))
    return clipped
