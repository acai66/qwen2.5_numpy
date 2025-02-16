import os

import numpy as np

def load_model(model_path: str) -> dict:
    results = dict()
    # print(os.path.exists(open(os.path.join(model_path, f'model_keys.txt'))))
    with open(os.path.join(model_path, f'model_keys.txt'), 'r', encoding='utf-8') as model_keys_txt:
        keys = model_keys_txt.readlines()
        for k in keys:
            k = k.strip()
            if k:
                npy_path = os.path.join(model_path, f'{k}.npy')
                results[k] = np.load(npy_path)

    return results


if __name__ == '__main__':
    model_weights = load_model('models/qwen2.5_0.5b_Instruct')
    print(model_weights.keys())
