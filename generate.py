import argparse
import math

import torch
from torchvision import utils

from model import EqualLinear, StyledGenerator


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

@torch.no_grad()

def sample(generator, step, mean_style, n_sample, device):
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image

@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device, style_weight=0.7, mixing_range=(0, 1)):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=style_weight,
            mixing_range=mixing_range,
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('path', type=str, help='path to checkpoint file')
    parser.add_argument('save', type=str, help='path to save outputs')
    
    args = parser.parse_args()
    
    device = 'cuda'

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()

    # for k, v in generator.generator.state_dict().items():
    #     print(k)
    idx = 0
    # temp = []
    prev = 16
    for s_i, c in enumerate(generator.generator.progression.children()):
    #     temp.append(c)
    # temp = temp[::-1]
    # print(temp[0])
    # for s_i, c in enumerate(temp):
        # print(type(c))
        # for a in c:
        print('\n{} -- {}th'.format(str(c).split('(')[0], s_i + 1))
        print("=" * 80) 
        # for a1 in a.children():
        
        for a1 in c.children():
            if 'model' in str(type(a1)) or 'Sequential' in str(type(a1)):
                # print(a1.children())
                if 'model' in str(type(a1)):
                    print('[ {} ]'.format(str(a1).split('(')[0]))
                for a2 in a1.children(): # designed module
                    if 'EqualLinear' in str(a2) or 'EqualConv2d' in str(a2):
                        toprint = str(a2).split(',')
                        name, *_, in_ = toprint[0].split('(')
                        in_c = in_.replace('in_features=', '')
                        out_c = toprint[1].replace('out_features=', '')
                        name = name.replace(',', '')
                        in_c = in_c.replace(',', '')
                        if 'EqualLinear' in str(a2):
                            if 'AdaptiveInstanceNorm' in str(a1):
                                print('{:>20} {:>25} {:>15} {:>15} {:>15} {:>15}, {}'.format(name, in_c, '-->', out_c, '-->', int(out_c)//2, int(out_c)//2))
                            else:
                                print('{:>20} {:>25} {:>15} {:>15} {:>15} {:>15}, {}'.format(name, in_c, '-->', out_c, '-->', int(out_c)//2, int(out_c)//2))
                                
                            prev = int(out_c) // 2
                        else:
                            print('{:>20} {:>25} {:>15} {:>15} '.format(name, in_c, '-->', out_c))
                            prev = int(out_c)
                    elif 'Conv2d' in str(a2):
                        toprint = str(a2).split('(')
                        name = toprint[0]
                        in_c, out_c = toprint[1].split(',')[:2]
                        name = name.replace(',', '')
                        in_c = in_c.replace(',', '')
                        print('{:>20} {:>25} {:>15} {:>15}'.format(name, in_c, '-->', out_c))
                    elif 'InstanceNorm2d' in str(a2):
                        toprint = str(a2).split('(')
                        name, ch = toprint[0], toprint[1].split(',')[0]
                        print('{:>20} {:>25} {:>15} {:>15}'.format(name, ch, '-->', ch))
                        # print(str(a2))
                    elif 'FusedUpsample' in str(a2) or 'Upsample' in str(a2):
                        if 'FusedUpsample' in str(a2):
                            in_c, out_c , *_ = a2.weight.shape
                            print('{:>20} {:>25} {:>15} {:>15}'.format(str(a2).split('(')[0], in_c, '-->', out_c))
                        else:
                            print('{:>20} {:>36}'.format(str(a2).split('(')[0], 'Keep Dimension'))
                            
                    else:
                        print('{:>20}'.format(str(a2).split('(')[0]))

                    idx += 1
            else: # nn.module
                if 'Conv2d' in str(a1):
                    toprint = str(a1).split('(')
                    name = toprint[0]
                    in_, out, *_ = toprint[1].split(',')[:2]
                    name = name.replace(',', '')
                    in_ = in_c.replace(',', '')
                    print('{:>20} {:>25} {:>15} {:>15}'.format(name, in_, '-->', out))
                else:
                    print('{:>20}'.format(str(a1).split('(')[0]))
                idx += 1
        print("=" * 80) 

                # else:
                #     print(type(a1))
   
    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2
    
    img = sample(generator, step, mean_style, args.n_row * args.n_col, device)
    utils.save_image(img, 'sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    
    # for j in range(20):
    #     img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
    #     utils.save_image(
    #         img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
    #     )

    import os
    os.makedirs(args.save, exist_ok=True)

    test_mixing_ranges = [
        (0, 1),
        (2, 3),
        (4, 8)]

    # for j in range(2): # Style weight 
    #     for i in range(10 + 1):
    #         print(f'Processing: {args.save}/sample_mixing_{j}_0.{i}.png')
    #         img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device,
    #             style_weight=i/10)
    #         utils.save_image(
    #             img, f'{args.save}/sample_mixing_{j}_0.{i}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
    #         )2, 3


    # for j in range(2): # mixing range 0 ~ N
    #     for i in range(10 + 2):
    #         print(f'Processing: {args.save}/sample_mixing_{j}_mx_0,{i}.png')
    #         img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device,
    #             mixing_range=(0, i))
    #         utils.save_image(
    #             img, f'{args.save}/sample_mixing_{j}_mx_0,{i}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
    #         )

    for j in range(2): # test mixing range
        for mixing_ranges in test_mixing_ranges:
            print(f'Processing: {args.save}/sample_mixing_{mixing_ranges[-1]}.png')
            img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device,
                mixing_range=mixing_ranges)
            utils.save_image(
                img, f'{args.save}/sample_mixing_{mixing_ranges[-1]}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
            )


    # for j in range(2): # N ~ end mixing range
    #     for i in range(10 + 2):
    #         print(f'Processing: {args.save}/sample_mixing_{j}_mx_0,{i}.png')
    #         img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device,
    #             mixing_range=(0, i))
    #         utils.save_image(
    #             img, f'{args.save}/sample_mixing_{j}_mx_0,{i}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
    #         )


    # for j in range(2):
    #     for i in range(10 + 2):
    #         print(f'Processing: {args.save}/sample_mixing_{j}_{i},9.png')
    #         img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device,
    #             mixing_range=(i, 9))
    #         utils.save_image(
    #             img, f'{args.save}/sample_mixing_{j}_mx_{i},9.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
    #         )