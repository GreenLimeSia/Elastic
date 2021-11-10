# python 3.6
"""Demo."""

import numpy as np
import torch
from op.id_loss import IDLoss
import streamlit as st
from utils import SessionState
import pandas as pd
from glob import glob
from models import parse_gan_type
from utils.utils import to_tensor, postprocess, load_generator, factorize_weight_SeFa

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# @st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    return load_generator(model_name)


# @st.cache(allow_output_mutation=True, show_spinner=False)
def factorize_model(model, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight_SeFa(model, layer_idx)


def sample(model, gan_type, num=1):
    """Samples latent codes."""
    codes = torch.randn(num, model.z_space_dim).to(device)
    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=18)
    codes = codes.detach().cpu().numpy()
    return codes


# @st.cache(allow_output_mutation=True, show_spinner=False)
def synthesize(model, gan_type, code):
    """Synthesizes an image with the give code."""
    if gan_type == 'pggan':
        image = model(to_tensor(code))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        image = model.synthesis(to_tensor(code))['image']
    return image


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


def euclidean_distance(a, b):
    euclidean_dist = np.sqrt(np.square(a - b).sum())
    return euclidean_dist


def main():
    """Main function (loop for StreamLit)."""
    st.title('An Adaptive Weight Modulation Method for Facial Image Editing')
    st.sidebar.title('Options')
    reset = st.sidebar.button('Reset')

    model_name = st.sidebar.selectbox(
        'Model to Interpret',
        ['stylegan2_ffhq1024', 'stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256',
         'pggan_celebahq1024'])

    model, _ = get_model(model_name)
    gan_type = parse_gan_type(model)
    layer_idx = st.sidebar.selectbox(
        'Layers to Interpret',
        ['all', '0-1', '2-5', '6-13'])
    # layers, U, S, V = factorize_model(model, layer_idx)
    layers, boundaries, eigen_values = factorize_model(model, layer_idx)
    direction_name = ['pose', 'age', 'glasses', 'surprise', 'sad']

    num_semantics = st.sidebar.number_input(
        'Number of semantics', value=5, min_value=0, max_value=None, step=1)
    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    spec_attr = sorted([filename for filename in glob(r'./latent_directions/*.npy')])
    spec_attr = spec_attr[:num_semantics]
    if gan_type == 'pggan':
        max_step = 5.0
    elif gan_type == 'stylegan':
        max_step = 2.0
    elif gan_type == 'stylegan2':
        max_step = 14.49
    # for sem_idx in steps:
    #     # eigen_value = eigen_values[sem_idx]
    #
    #     eigen_value = eigen_values[sem_idx]
    #     steps[sem_idx] = st.sidebar.slider(
    #         f'Semantic {sem_idx:03d} (eigen value: {eigen_value:.3f})',
    #         value=0.0,
    #         min_value=-max_step,
    #         max_value=max_step,
    #         step=0.04 * max_step if not reset else 0.0)

    for sem_idx in steps:
        # eigen_value = eigen_values[sem_idx]

        # eigen_value = np.load(spec_attr[sem_idx])
        steps[sem_idx] = st.sidebar.slider(
            f'Semantic {sem_idx:03d} (direction_name: {direction_name[sem_idx]})',
            value=0.0,
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step if not reset else 0.0)

    image_placeholder, image_placeholder_temp = st.beta_columns(2)
    # image_placeholder = st.empty()
    button_placeholder = st.empty()
    table_data = st.empty()

    try:
        base_codes = np.load(f'latent_codes/{model_name}_latents.npy')
    except FileNotFoundError:
        base_codes = sample(model, gan_type)

    state = SessionState.get(model_name=model_name,
                             code_idx=0,
                             codes=base_codes[0:1])
    if state.model_name != model_name:
        state.model_name = model_name
        state.code_idx = 0
        state.codes = base_codes[0:1]

    # define identity loss
    id_loss = IDLoss().to(device).eval()

    if button_placeholder.button('Random', key=0):
        state.code_idx += 1
        if state.code_idx < base_codes.shape[0]:
            state.codes = base_codes[state.code_idx][np.newaxis]
        else:
            state.codes = sample(model, gan_type)

    code, temp_code = state.codes.copy(), state.codes.copy()

    # for sem_idx, step in steps.items():
    #     if gan_type == 'pggan':
    #         code += boundaries[sem_idx:sem_idx + 1] * step
    #     elif gan_type in ['stylegan', 'stylegan2']:
    #         code[:, layers, :] += boundaries[sem_idx:sem_idx + 1] * step

    for sem_idx, step in steps.items():
        if gan_type == 'pggan':
            code += boundaries[sem_idx:sem_idx + 1] * step
        elif gan_type in ['stylegan', 'stylegan2']:
            boundaries = np.load(spec_attr[sem_idx])
            code[:, layers, :] += boundaries[layers, :] * step
            # code[:, layers, :] += boundaries[sem_idx:sem_idx + 1] * step

    raw_image = synthesize(model, gan_type, temp_code)
    image = synthesize(model, gan_type, code)

    iden_loss, _, _ = id_loss(image, raw_image, raw_image)

    _, h, w = code.shape
    cos_dist = cosine_distance(code.reshape(h * w), temp_code.reshape(h * w)).item()
    euc_distance = euclidean_distance(code, temp_code)
    image_placeholder.image(postprocess(image)[0] / 255.0, width=330, caption="Editing")

    image_placeholder_temp.image(postprocess(raw_image)[0] / 255.0, width=330, caption="Original")
    # image_placeholder.image([postprocess(image)[0] / 255.0, postprocess(raw_image)[0] / 255.0], use_column_width=True,
    #                         caption=["Editing", "Original"])

    table_data.header('A dynamic Score for Editing')
    table_data.write(pd.DataFrame({
        'Identity score': [1 - iden_loss.item()],
        #'Accuracy': 0.5,
        #'Error': 0.5,
        'Cosine distance': [cos_dist],
        'Euclidean distance': [euc_distance]
    }))


if __name__ == '__main__':
    main()
