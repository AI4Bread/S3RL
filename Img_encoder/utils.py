from PIL import Image

class PatchDataset:
    def __init__(self, patchs, transform) -> None:
        self.patchs = [Image.fromarray(patch) for patch in patchs]
        self.trasform = transform

    def __len__(self):
        return len(self.patchs)
    
    def __getitem__(self, index):
        if self.trasform is not None:
            return self.trasform(self.patchs[index]), self.trasform(self.patchs[index])