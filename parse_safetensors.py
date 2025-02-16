import os
import sys
import struct
import json
import argparse
import shutil

import numpy as np

from functools import reduce


def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b

def q8_0_quantize(data: np.ndarray, block_size:int=32) -> np.ndarray:
    """Q8_0量化，来自gguf"""
    original_shape = data.shape
    blocked_data = data.reshape(-1, block_size) # 每 32 个元素为一块，块内计算量化因子
    d = np.abs(blocked_data).max(axis=-1, keepdims=True) / 127 # 间隔 d ，8位量化时：|最大值| / (2**(8-1)-1)
    with np.errstate(divide="ignore"):
        scale = np.where(d == 0, 0, 1 / d) # 量化因子，间隔 d 的倒数
    quant_data = np_roundf(blocked_data * scale) # 量化：除以间隔，四舍五入取整，此时数据类型为浮点数，数值为-127~127的整数
    quant_data = quant_data.astype(np.int8) # 转换为 int8 类型，减小数据的大小

    return quant_data.reshape(original_shape), d # d 为反量化因子，量化时除以 d，反量化时乘以 d


def bf16_to_fp32(bf16_bytes: bytes, shape: list) -> np.ndarray:
    """将BF16字节数据转换为FP32 numpy数组"""
    # 将字节转换为uint16数组
    dt = np.dtype(np.uint16)
    # dt = dt.newbyteorder('>') # 无需调整大小端
    uint16_arr = np.frombuffer(bf16_bytes, dtype=dt)
    # 转换为uint32并左移16位，然后通过view方法转为float32
    fp32_arr = (uint16_arr.astype(np.uint32) << 16).view(np.float32) #.astype(np.float16)
    # 调整形状
    return fp32_arr.reshape(shape)


def parse_safetensors_header(file_dir, npy_save_dir, quantize=False):
    shutil.copyfile(os.path.join(file_dir, 'config.json'), os.path.join(npy_save_dir, 'config.json'))
    shutil.copyfile(os.path.join(file_dir, 'tokenizer.json'), os.path.join(npy_save_dir, 'tokenizer.json'))
    safetensors_files = [f for f in os.listdir(file_dir) if f.endswith('.safetensors')]
    print(f'找到SafeTensors文件: {safetensors_files}')

    with open(os.path.join(npy_save_dir, 'model_keys.txt'), 'w+', encoding='utf-8') as model_keys_txt:
        for file in safetensors_files:
            file_path = os.path.join(file_dir, file)
            with open(file_path, 'rb') as f:
                # 读取头部长度（前8字节，小端uint64）
                header_len_bytes = f.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]

                # 读取头部JSON数据（根据header_len指定的长度）
                header_bytes = f.read(header_len)
                header_len += 8
                
                # 解析JSON
                header_str = header_bytes.decode('utf-8')
                header = json.loads(header_str)

                for k in header:
                    if k != '__metadata__':
                        dtype = header[k]['dtype']
                        if dtype != 'BF16':
                            print(f"错误: 不支持的类型: {dtype}，退出")
                            sys.exit(-11)
                        shape = header[k]['shape']
                        if len(shape) > 2:
                            print(f"错误: 不支持的形状: {shape}，退出")
                            sys.exit(-12)
                        data_offsets = header[k]['data_offsets']
                        if data_offsets[1] - data_offsets[0] != reduce(lambda x,y:x*y, shape) * 2:
                            print(f"错误: 不支持的形状和偏移量: {shape} {data_offsets[0]}-{data_offsets[1]}，退出")
                            sys.exit(-13)

                        print(f'key: {k}, dtype: {dtype}, shape: {shape}, data_offsets: {data_offsets}')
                        f.seek(header_len + data_offsets[0])
                        data = bf16_to_fp32(f.read(data_offsets[1] - data_offsets[0]), shape) # 转 fp32
                        if k.endswith('.weight') and (not k.endswith('embed_tokens.weight')):
                            data = data.T
                        # 量化，排除 norm 层和 bias 偏置，因为这些层数据量不大，量化后误差增大，没有意义
                        if quantize and k.endswith('.weight') and (not k.endswith('norm.weight')):
                            data, dequantize = q8_0_quantize(data)
                            np.save(os.path.join(npy_save_dir, f'{k}_dequantize.npy'), dequantize) # 保存反量化因子
                            model_keys_txt.write(f'{k}_dequantize\n')
                        np.save(os.path.join(npy_save_dir, f'{k}.npy'), data)
                        model_keys_txt.write(f'{k}\n')
                if 'model.embed_tokens.weight' in header and'model.embed_tokens.weight' not in header:
                    print(f"错误: 未知的键值: {header['']}")
                    sys.exit(-14)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='解析SafeTensors模型文件，保存为Numpy float32数组')
    parser.add_argument('--model_dir', type=str, help='SafeTensors模型文件路径')
    parser.add_argument('--npy_save_dir', type=str, help='Numpy数组保存路径')
    parser.add_argument('--quantize', action='store_true', help='是否量化，当前仅支持Q8_0')

    args = parser.parse_args()

    model_dir = args.model_dir
    npy_save_dir = args.npy_save_dir
    quantize = args.quantize

    os.makedirs(npy_save_dir, exist_ok=True)

    parse_safetensors_header(model_dir, npy_save_dir, quantize)
