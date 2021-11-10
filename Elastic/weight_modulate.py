"""Discover sub-generator generating and editing space"""
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import copy
# from PIL import Image
import pandas as pd
from torchvision import utils
from models import parse_gan_type
from utils.utils import to_tensor
# from utils.utils import postprocess
from utils.utils import load_generator, load_generator_custom
from utils.utils import factorize_weight
from sewar.full_ref import mse, rmse, uqi, vifp
from piq import psnr
from pytorch_msssim import ssim, ms_ssim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('--model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--window', type=int, default=5,
                        help='sliding window for updating. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-E', '--epoch', type=int, default=100,
                        help='Number of epoch used for generation. '
                             '(default: %(default)s)')
    parser.add_argument('-V', '--visualize', type=int, default=50,
                        help='Number of epoch used for generation. '
                             '(default: %(default)s)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU(s) to use. (default: %(default)s)')
    parser.add_argument('--custom', action='store_true', default=False)
    parser.add_argument("--mode", type=str, default='baby')
    return parser.parse_args()


def mask_lamda(eigenvalue, index):
    mask = torch.ones(eigenvalue.shape[0]).to(device)
    mask[index] = 0
    eigenvalue = eigenvalue * mask
    return eigenvalue


def sample(model, gan_type, num=1, latent_dim=512):
    """Samples latent codes."""
    codes = torch.randn(num, latent_dim).to(device)
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
        return codes

    codes = codes.detach().cpu().numpy()
    return codes


def synthesize(model, gan_type, code):
    """Synthesizes an image with the give code."""
    image = None
    if gan_type == 'pggan':
        image = model(to_tensor(code))['image']
    if gan_type in ['stylegan', 'stylegan2']:
        image = model.synthesis(to_tensor(code))['image']
    if gan_type is None:
        image, _ = model(
            [code], truncation=1.0, truncation_latent=4096
        )

    # save image used torch directly and remove the following code
    # image = postprocess(image)
    return image


def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(0, 2, 3, 1).cpu().numpy()
    return tensor * 0.5 + 0.5


def make_dir(mode):
    os.makedirs("results_raw_{}".format(mode), exist_ok=True)
    os.makedirs('results_update_{}'.format(mode), exist_ok=True)


def update_generator(U, S, V, split_list, modulate, index, model_name, custom):
    S = mask_lamda(S, index=[i for i in range(index, 512)])

    S_dia = torch.diag(S)

    # print(S_dia.shape)

    U_update = U.mm(S_dia).mm(V.T)

    # print('U_update.shape', U_update.shape) (8192, 512) PGGAN

    U_uncat = torch.split(U_update, split_list, dim=0)

    # print("U_uncat[-1]", U_uncat[-1])

    update_key = [k for k in modulate.keys()]

    updated_dict = dict(zip(update_key, U_uncat))

    # print('updated_dict', updated_dict['layer0.weight'].shape)

    if custom:
        generator_update, _ = load_generator_custom(model_name, update_weights=updated_dict)
        print("Using a custom updated Generator")
    else:
        generator_update, _ = load_generator(model_name, update_weights=updated_dict)
        print("Using an updated Generator")

    return generator_update


def main():
    """Main function."""
    global index, split_list, modulate, U, V, S, gan_type
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.mode == 'baby':
        make_dir(args.mode)

    if args.mode == 'star':
        make_dir(args.mode)

    if args.mode == 'wanghong':
        make_dir(args.mode)

    if args.mode == 'yellow':
        make_dir(args.mode)

    if args.mode == 'chaomo':
        make_dir(args.mode)

    if args.mode == 'official':
        make_dir(args.mode)

    if args.mode == 'stylegan':
        make_dir(args.mode)

    if args.mode == 'pggan':
        make_dir(args.mode)

    if args.custom:
        generator, modulate = load_generator_custom(args.model_name)
        split_list = list(modulate[k].shape[0] for k, _ in modulate.items())
        # print('split_list', split_list) split_list [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32]
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

        # print(split_list)

        if args.mode == 'pggan':
            split_list = [8192]

        ############################################################

        gan_type = parse_gan_type(generator)
        layers, U, S, V = factorize_weight(generator, args.layer_idx)

    index = [int(i) for i in np.linspace(args.window, V.shape[0], V.shape[0] // args.window)]

    # resume previous results
    # if args.mode == 'pggan':
    #     index = index[index.index(125):]
    #     print("Load index:", index.index(125))

    temp_mse, temp_rmse, temp_uqi, temp_ssim, temp_psnr, temp_vif, temp_mssim = float('inf'), float(
        'inf'), 0., 0., 0., 0., 0.

    with torch.no_grad():
        try:
            for index_id in index:

                ssim_list, psnr_list, vif_list, mssim_list, mse_list, rmse_list, uqi_list = [list() for x in range(7)]

                generator_update = update_generator(U, S, V, split_list, modulate, index_id, args.model_name,
                                                    args.custom)
                # print(generator_update.state_dict()['synthesis.layer0.style.weight'])
                # print(generator_update.state_dict()['synthesis.layer10.style.weight'])

                for epoch in tqdm(range(args.epoch)):
                    # Prepare codes.
                    codes = sample(generator, gan_type, num=args.num_samples)

                    image_raw = synthesize(generator, gan_type, codes)
                    image_update = synthesize(generator_update, gan_type, codes)
                    # print(image.shape, type(image))  # torch.Size([4, 3, 1024, 1024])

                    mse_score = mse(tensor2image(image_raw.detach().cpu()), tensor2image(image_update.detach().cpu()))
                    rmse_score = rmse(tensor2image(image_raw.detach().cpu()), tensor2image(image_update.detach().cpu()))

                    # calculate uqi score
                    uqi_score = 0
                    for count in range(args.num_samples):
                        uqi_score += uqi(tensor2image(image_raw.detach().cpu())[count],
                                         tensor2image(image_update.detach().cpu())[count])
                    uqi_score = uqi_score / args.num_samples

                    # calculate ssim score

                    ssim_scor = ssim(image_raw.detach().cpu(), image_update.detach().cpu())

                    psnr_scor = psnr((image_raw.detach().cpu() + 1) / 2, (image_update.detach().cpu() + 1) / 2)

                    # print(imgs[0].transpose(2, 1, 0).shape)

                    vif_scor = 0
                    for count in range(args.num_samples):
                        vif_scor += vifp((image_raw.detach().cpu().numpy()[count].transpose(2, 1, 0) + 1) / 2,
                                         (image_update.detach().cpu().numpy()[count].transpose(2, 1, 0) + 1) / 2)

                    vif_scor = vif_scor / args.num_samples

                    mssim_scor = ms_ssim(image_raw.detach().cpu(), image_update.detach().cpu())

                    mse_list.append(mse_score.item())
                    rmse_list.append(rmse_score.item())
                    uqi_list.append(uqi_score.item())
                    ssim_list.append(ssim_scor.item())
                    psnr_list.append(psnr_scor.item())
                    vif_list.append(vif_scor.item())
                    mssim_list.append(mssim_scor.item())

                    print(mse_score.item(), rmse_score.item(), uqi_score.item(), ssim_scor.item(), psnr_scor.item(),
                          vif_scor.item(), mssim_scor.item())

                    if epoch % args.visualize == 0:
                        # with torch.no_grad():
                        #     generator.eval()
                        #     generator_update.eval()
                        for i in tqdm(range(args.num_samples)):
                            #     im = Image.fromarray(image[i])
                            #     im.save(f"results/{str(i).zfill(3)}.png")

                            utils.save_image(
                                image_raw[i],
                                "results_raw_{}/{}.png".format(args.mode,
                                                               str(str(i).zfill(3)) + " " + str(index_id) + " " + str(
                                                                   epoch)),
                                nrow=1,
                                normalize=True,
                                range=(-1, 1),
                            )

                            utils.save_image(
                                image_update[i],
                                "results_update_{}/{}.png".format(args.mode,
                                                                  str(str(i).zfill(3)) + " " + str(
                                                                      index_id) + " " + str(epoch)),
                                nrow=1,
                                normalize=True,
                                range=(-1, 1),
                            )

                # save all metric data
                data = {'mse': mse_list, 'rmse': rmse_list, 'uqi': uqi_list, 'ssim': ssim_list, 'psnr': psnr_list,
                        'vif': vif_list, 'mssim': mssim_list}

                df = pd.DataFrame(data)

                df.to_csv('./results_raw_{}/{}.csv'.format(args.mode, str(args.epoch) + " " + str(index_id)))

                mean_mse, mean_rmse, mean_uqi, mean_ssim, mean_psnr, mean_vif, mean_mssim = [
                    float(format(np.mean(value), '.6f')
                          ) for _, value in data.items()]

                std_mse, std_rmse, std_uqi, std_ssim, std_psnr, std_vif, std_mssim = [
                    float(format(np.std(value), '.6f')
                          ) for _, value in data.items()]

                # we use average score as the metrics, convergence condition
                if (mean_mse + std_mse) > temp_mse and (mean_rmse + std_rmse) > temp_rmse \
                        and (mean_uqi - std_uqi) < temp_uqi and (mean_ssim - std_mssim) < temp_ssim \
                        and (mean_psnr - std_psnr) < temp_psnr and (mean_vif - std_vif) < temp_vif \
                        and (mean_mssim - std_mssim) < temp_mssim:
                    break

                temp_mse, temp_rmse, temp_uqi, temp_ssim, temp_psnr, temp_vif, temp_mssim = mean_mse, mean_rmse, mean_uqi, mean_ssim, mean_psnr, mean_vif, mean_mssim
                # ssim_list.clear(), psnr_list.clear(), vif_list.clear(), mssim_list.clear(), mse_list.clear(),
                # rmse_list.clear(), uqi_list.clear()

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception


if __name__ == '__main__':
    main()
