import torch
import numpy as np
from glob import glob


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

    # normalize distance to range of [0,1]
    dist = (1. - similiarity) / 2.0
    return dist


def euclidean_distance(a, b):
    euclidean_dist = np.sqrt(np.square(a - b).sum())
    return euclidean_dist


if __name__ == '__main__':
    direction_name = ['age', 'pose', 'glasses', 'smile', 'gender']

    V_semantic = torch.load(r'./semantic.pt')['boundaries'].T

    num_sem = len(direction_name)
    spec_attr = sorted([filename for filename in glob(r'./attribute_code/*.npy')])
    spec_attr = spec_attr[:num_sem]

    print(spec_attr)
    for supervised_semantic_id in range(num_sem):
        super_sem = np.load(spec_attr[supervised_semantic_id]).reshape(18 * 512)
        for sem_id in range(num_sem):
            unsuper_sem = V_semantic[sem_id:sem_id + 1].repeat(18, axis=0).reshape(18 * 512)
            cos_distance = cosine_distance(super_sem, unsuper_sem)
            euc_distance = euclidean_distance(super_sem, unsuper_sem)
            print(direction_name[supervised_semantic_id], 'unsuper_sem_{}'.format(sem_id), 'cosine_distance', cos_distance)
            print(direction_name[supervised_semantic_id], 'unsuper_sem_{}'.format(sem_id), 'euc_distance', euc_distance)
