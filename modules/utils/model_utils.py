from torch import nn

def print_trainable_parameters(model: nn.Module) -> None:
    parameters, trainable = 0, 0
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    print(f"trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.1f}%)")