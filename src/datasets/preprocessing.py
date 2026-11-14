import torch
import torchvision.transforms as T


class Preprocessor:
    def __init__(self, size: int, mean, std):
        self.mean = mean
        self.std = std

        self.pipeline = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            self.mean, device=x.device
        ).view(3, 1, 1)

        std = torch.tensor(
            self.std, device=x.device
        ).view(3, 1, 1)

        return (x * std + mean).clamp(0, 1)