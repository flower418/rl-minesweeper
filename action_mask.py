import torch


def apply_action_mask(logits, mask, invalid_fill=-1e9):
    mask = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)

    if logits.dim() == 2 and mask.dim() == 1:
        mask = mask.unsqueeze(0).expand_as(logits)
    elif logits.shape != mask.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match logits shape {tuple(logits.shape)}"
        )

    return logits.masked_fill(~mask, invalid_fill)
