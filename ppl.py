import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import lpips
from model import StyledGenerator

import math

def make_noise(device,size):
    noises = []
    step = size - 2
    for i in range(size + 1):
            size = 4 * 2 ** i
            noises.append(torch.randn(1, 1, size, size, device=device))
    return noises

# def make_noise(device, log_size):

#     noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

#     for i in range(3, log_size + 1):
#         for _ in range(2):
#             noise = torch.randn(1, 1, 2 ** i, 2 ** i, device=device)
#             noises.append(noise)
#             print(noise.shape)
#     return noises

def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))

@torch.no_grad()
def get_mean_style(generator, device, dim=1024):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(dim, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10

    return mean_style

def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Perceptual Path Length calculator")

    parser.add_argument(
        "--space", choices=["z", "w"], help="space that PPL calculated with"
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the models"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5000,
        help="number of the samples for calculating PPL",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-4, help="epsilon for numerical stability"
    )
    parser.add_argument(
        "--crop", action="store_true", help="apply center crop to the images"
    )
    parser.add_argument(
        "--sampling",
        default="end",
        choices=["end", "full"],
        help="set endpoint sampling method",
    )
    parser.add_argument(
        "ckpt", metavar="CHECKPOINT", help="path to the model checkpoints"
    )

    args = parser.parse_args()

    # Set embedding vector size to 512
    latent_dim = 512 

    # load checkpoint
    ckpt = torch.load(args.ckpt)
    
    g = StyledGenerator(512).to(device)
    g.load_state_dict(ckpt['g_running'])
    g.eval()

    step = int(math.log(args.size, 2)) - 2

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )   

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch# + [resid]

    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            noise = make_noise(device, step)

            inputs = torch.randn([batch * 2, latent_dim], device=device)
            if args.sampling == "full":
                lerp_t = torch.rand(batch, device=device)
            else:
                lerp_t = torch.zeros(batch, device=device)

            if args.space == "w":
                latent = g.style(inputs)
                latent_t0, latent_t1 = latent[::2], latent[1::2]
                # latent_t0, latent_t1 = f(latent_t0), f(latent_t1)
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

                image = g([latent_e], noise=noise, step=step)
            else:
                latent_t0, latent_t1 = inputs[::2], inputs[1::2]
                latent_e0 = slerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = slerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*inputs.shape)
                image = g([latent_e], noise=noise, step=step)

            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                args.eps ** 2
            )
            # if 0 in dist:
            #     continue
            distances.append(dist.to("cpu").numpy())

    distances = np.concatenate(distances, axis=0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print("ppl:", filtered_dist.mean())
