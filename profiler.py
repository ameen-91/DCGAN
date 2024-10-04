import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


model = Generator(ngpu=0)

model.load_state_dict(torch.load("car_gen.pth"))


def generate(button):

    model.eval()

    noise = torch.randn(32, 128, 1, 1)

    with torch.inference_mode():
        images = []
        predictions = model(noise).detach().cpu()
        generated_grid = vutils.make_grid(
            predictions, nrow=8, padding=2, normalize=True
        )
    return to_pil_image(generated_grid)
