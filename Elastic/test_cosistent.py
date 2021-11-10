"""Models form SeFa. Thanks for their excellent work."""

import os
import argparse
from tqdm import tqdm
import numpy as np
from torchsummaryX import summary
from torchvision import utils
import torch
from glob import glob
from models import parse_gan_type
from utils.utils import to_tensor
from utils.utils import postprocess
from utils.utils import load_generator, load_generator_custom
from utils.utils import factorize_weight
from utils.utils import HtmlPageVisualizer
from op.id_loss import IDLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('--model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--attribute_name', type=str,
                        help='Name to the attribute.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('--code_dir', type=str, default='latent_codes',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=2,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('-I', '--index_id', type=int, default=90)
    parser.add_argument('--start_distance', type=float, default=-3.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=3.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=15,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Size of images to visualize on the HTML page. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--save_period', type=int, default=50,
                        help='Save temp results. (default: %(default)s)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU(s) to use. (default: %(default)s)')
    parser.add_argument('--custom', action='store_true', default=False)
    parser.add_argument('--specific_attr', action='store_true', default=False)
    return parser.parse_args()


def mask_lamda(eigenvalue, index):
    mask = torch.ones(eigenvalue.shape[0]).to(device)
    mask[index] = 0
    eigenvalue = eigenvalue * mask
    return eigenvalue


def sample(model, gan_type, num=1, latent_dim=512):
    """Samples latent codes."""
    codes = torch.randn(num, latent_dim, device=device, dtype=torch.float32)
    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    if gan_type == 'stylegan':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    if gan_type == 'stylegan2':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=18)
    if gan_type is None:
        codes = model.get_latent(codes)  # get style codes via affine transform
        codes = codes.view(-1, 1, latent_dim)
        codes = codes.repeat(1, 18, 1)

    # print(codes.shape)(300, 18, 512)

    codes = codes.detach().cpu().numpy()
    return codes


def synthesize(model, gan_type, codes):
    """Synthesizes an image with the give code."""
    image = None
    with torch.no_grad():
        model.eval()
        if gan_type == 'pggan':
            image = model(to_tensor(codes))['image']
        if gan_type in ['stylegan', 'stylegan2']:
            image = model.synthesis(to_tensor(codes))['image']
        if gan_type is None:
            codes = torch.tensor(codes, device=device)
            image, _ = model(
                [codes], input_is_latent=True,
                randomize_noise=False,
                return_latents=False
            )

    # save image used torch directly and remove the following code
    # image = postprocess(image)
    return image


def make_dir(model_name, semantic_name, ):
    os.makedirs("editing/{}/{}".format(model_name, semantic_name), exist_ok=True)


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))

    similiarity = np.dot(a, b.T) / (a_norm * b_norm)

    dist = 1. - similiarity
    return dist


def update_generator(U, S, V, split_list, modulate, index, model_name, custom):
    S = mask_lamda(S, index=[i for i in range(index, 512)])

    S_dia = torch.diag(S)

    # print(S_dia.shape)

    U_update = U.mm(S_dia).mm(V.T)

    U_uncat = torch.split(U_update, split_list, dim=0)

    # print("U_uncat[-1]", U_uncat[-1])

    update_key = [k for k in modulate.keys()]

    updated_dict = dict(zip(update_key, U_uncat))

    if custom:
        generator_update, _ = load_generator_custom(model_name, update_weights=updated_dict)
        print("Using a custom updated Generator")
    else:
        generator_update, _ = load_generator(model_name, update_weights=updated_dict)
        print("Using an updated Generator")

    return generator_update


def compute_cosine_distance():
    """Compute cosine distance."""
    global layers, split_list, modulate, U, V, S, gan_type
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)

    if args.custom:
        generator, modulate = load_generator_custom(args.model_name)
        split_list = list(modulate[k].shape[0] for k, _ in modulate.items())
        # print('split_list:', split_list)
        gan_type = None
        weight_mat = []
        for k, v in modulate.items():
            weight_mat.append(v)
        W = torch.cat(weight_mat, 0).to(device)
        U, S, V = torch.svd(W)

    else:
        # Factorize weights.
        generator, modulate = load_generator(args.model_name)

        # print(generator.state_dict()['synthesis.layer0.style.weight'])
        # print(generator.state_dict()['synthesis.layer10.style.weight'])

        split_list = list(modulate[k].shape[0] for k, _ in modulate.items())

        ############################################################

        gan_type = parse_gan_type(generator)
        layers, U, S, V = factorize_weight(generator, args.layer_idx)

    index_id = args.index_id

    generator_update = update_generator(U, S, V, split_list, modulate, index_id, args.model_name,
                                        args.custom)

    # Set random seed.
    # np.random.seed(args.seed)
    np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # Prepare codes randomly.
    codes = sample(generator, gan_type, num=args.num_samples)

    # Generate visualization pages.
    distances = np.array([10.0])
    num_sam = args.num_samples
    num_sem = args.num_semantics

    # direction_name = ['age', 'glasses', 'eye_open', 'gender', 'surprise']
    # direction_name = ['pose', 'age', 'glasses', 'surprise', 'sad']
    # direction_name = ['posing', 'aging']

    for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
        code = codes[sam_id:sam_id + 1]  # code.shape (1, 18, 512)
        # print(code.shape)
        spec_attr = sorted([filename for filename in glob(r'./latent_directions/*.npy')])
        spec_attr = spec_attr[:num_sem]

        make_dir(args.model_name, 'testing')

        boundary_1 = np.load(spec_attr[0])
        boundary_2 = np.load(spec_attr[1])
        editing_samples = []

        temp_code = code.copy()
        for id in range(4):
            if id == 0:
                if gan_type in ['stylegan', 'stylegan2', None]:
                    # every select layers should be add its corresponding boundary
                    if args.custom:
                        temp_code[0][:8] = (code[0] + 6.0 * boundary_1)[:8]
                    else:
                        temp_code[:, layers, :] += boundary_1[layers, :] * 10.0
                    image = synthesize(generator_update, gan_type, temp_code)
                    # print('image.shape', image.shape)
                    editing_samples.append(image)

            if id == 1:
                if gan_type in ['stylegan', 'stylegan2', None]:
                    # every select layers should be add its corresponding boundary
                    if args.custom:
                        temp_code[0][:8] = (code[0] + 6.0 * boundary_1 + 5.0 * boundary_2)[:8]
                    else:
                        temp_code[:, layers, :] += boundary_1[layers, :] * 10.0 + boundary_2[layers, :] * 15.0
                    image = synthesize(generator_update, gan_type, temp_code)
                    # print('image.shape', image.shape)
                    editing_samples.append(image)

            if id == 2:
                if gan_type in ['stylegan', 'stylegan2', None]:
                    # every select layers should be add its corresponding boundary
                    if args.custom:
                        temp_code[0][:8] = (code[0] + 5.0 * boundary_2)[:8]
                    else:
                        temp_code[:, layers, :] += boundary_2[layers, :] * 15.0
                    image = synthesize(generator_update, gan_type, temp_code)
                    # print('image.shape', image.shape)
                    editing_samples.append(image)

            if id == 3:
                if gan_type in ['stylegan', 'stylegan2', None]:
                    # every select layers should be add its corresponding boundary
                    if args.custom:
                        temp_code[0][:8] = (code[0] + 5.0 * boundary_2 + 6.0 * boundary_1)[:8]
                    else:
                        temp_code[:, layers, :] += boundary_1[layers, :] * 10.0
                    image = synthesize(generator_update, gan_type, temp_code)
                    # print('image.shape', image.shape)
                    editing_samples.append(image)

        editing_samples = torch.cat(editing_samples, dim=0)
        # if sam_id % args.save_period == 0:
        # for idx in range(len(editing_samples)):
        #     utils.save_image(
        #         editing_samples[idx],
        #         'editing/{}/{}/{}_{}.png'.format(args.model_name, direction_name[sem_id], sam_id, idx),
        #         nrow=5,  # len(distances),
        #         normalize=True,
        #         range=(-1, 1),
        #     )
        utils.save_image(
            editing_samples,
            'editing/{}/{}/{}.png'.format(args.model_name, 'testing', sam_id),
            nrow=5,  # len(distances),
            normalize=True,
            range=(-1, 1),
        )


if __name__ == '__main__':
    # compute cosine distance and save its editing image with compute_cosine_distance function
    compute_cosine_distance()

'''python3 max_slope.py --custom --specific_attr --start_distance -30 --end_distance 30 --num_semantics 5 
--model_name stylegan2-baby --save_period 100 --step 30 -N 2000 -I 90 '''
