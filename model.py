import os
import json
import time

import numpy as np

from tokenizers import Tokenizer # tokenizers不依赖于torch、transformers等库

from load_npy_model import load_model

DTYPE = np.float32 # numpy 底层 BLAS 只支持 float32、float64 的矩阵加速运算，float16、整形等类型效率极低，需从底层代码优化

def rotate_half(x: np.ndarray) -> np.ndarray:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate((-x2, x1), axis=-1)

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    max_vals = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - max_vals)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x

def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))

def repeat_kv(hidden_states: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = np.broadcast_to(hidden_states[:, :, None, :, :], (batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def logits_processor(input_ids, scores, penalty=1.1, temperature=0.7, top_k=20, filter_value=float("-inf"), top_p=0.8, min_tokens_to_keep=1):
    """后处理，在随机采样前对logits值进行处理，包括重复惩罚、温度调整、top-k、top-p等"""
    # RepetitionPenaltyLogitsProcessor
    score = np.take_along_axis(scores, input_ids, 1)
    score = np.where(score < 0, score * penalty, score / penalty)
    np.put_along_axis(scores, input_ids, score, 1)

    # TemperatureLogitsWarper
    scores = scores / temperature

    # TopKLogitsWarper
    top_k = min(top_k, scores.shape[-1])  # Safety check
    indices_to_remove_np = scores < np.partition(scores, -top_k)[..., -top_k, None]
    scores[indices_to_remove_np] = filter_value

    # TopPLogitsWarper
    sorted_indices = np.argsort(scores)
    sorted_logits = np.take_along_axis(scores, sorted_indices, -1)
    cumulative_probs = softmax(sorted_logits).cumsum(axis=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0
    np.put_along_axis(sorted_indices_to_remove, sorted_indices, sorted_indices_to_remove, 1)
    scores[sorted_indices_to_remove] = filter_value

    return scores

def get_weights(model_weights: dict, weight_name: str) -> tuple:
    """从字典中获取权重矩阵和反量化参数"""
    weights = model_weights[weight_name]
    dequantize = model_weights.get(f'{weight_name}_dequantize', None)
    if dequantize is not None:
        dequantize = dequantize.astype(DTYPE)

    return weights, dequantize

def dequantize(weights: np.ndarray, d: np.ndarray = None) -> np.ndarray:
    """反量化权重为 float32"""
    if d is None:
        return weights
    original_shape = weights.shape
    # 实时反量化为 float32，然后在 float32 上进行矩阵乘法，相比于直接在 float32 上进行矩阵乘法，这里多了一步反量化操作，耗时更长，但是可以减少权重的内存占用(fp32权重->int8权重)
    weights = weights.reshape(-1, 32).astype(d.dtype) * d
    return weights.reshape(original_shape)


class DynamicCache:
    """动态缓存，用于存储每层的 key 和 value"""
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(
        self,
        key_states : np.ndarray,
        value_states: np.ndarray,
        layer_idx: int,
    ) -> tuple:
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # 如果层号不从0开始，或者不连续，跳过这些缺失的层号
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # 如果先前跳过的层号又出现了，用这次的数据替换跳过时填充的空列表
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else: # 常规更新，追加数据，创建新矩阵
                self.key_cache  [layer_idx] = np.concat([self.key_cache  [layer_idx], key_states  ], axis=-2)
                self.value_cache[layer_idx] = np.concat([self.value_cache[layer_idx], value_states], axis=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx = 0) -> int:
        is_empty_layer = (
            len(self.key_cache) == 0  # 未缓存任何层
            or len(self.key_cache) <= layer_idx  # 跳过的层，没缓存到的层
            or len(self.key_cache[layer_idx]) == 0  # 该层未缓存任何数据
        )
        return self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0


class Qwen2RMSNorm():
    def __init__(self, weight: np.ndarray, eps: float=1e-6):
        self.weight = weight
        self.variance_epsilon = eps

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        variance = np.pow(hidden_states, 2).mean(-1, keepdims=True)
        hidden_states = hidden_states / np.sqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class Qwen2RotaryEmbedding():
    def __init__(self, config: dict):
        self.max_seq_len_cached = config['max_position_embeddings']
        self.original_max_seq_len = config['max_position_embeddings']

        base = config['rope_theta']
        partial_rotary_factor = config.get("partial_rotary_factor", 1.0)
        head_dim = config.get("head_dim", config['hidden_size'] // config['num_attention_heads'])
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / dim))

    def __call__(self, x: np.ndarray, position_ids: np.ndarray) -> tuple:
        # Core RoPE block
        inv_freq_expanded = np.broadcast_to(self.inv_freq[None, None, :], (position_ids.shape[0], 1, self.inv_freq.shape[0]))
        position_ids_expanded = position_ids[:, :, None]

        freqs = (position_ids_expanded @ inv_freq_expanded)
        cos = np.cos(freqs)[:, None].astype(x.dtype)
        sin = np.sin(freqs)[:, None].astype(x.dtype)
        cos = np.concatenate((cos, cos), axis=-1)
        sin = np.concatenate((sin, sin), axis=-1)

        return cos, sin


class Qwen2Attention():
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: dict, model_weights: dict, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.get("head_dim", config['hidden_size'] // config['num_attention_heads'])
        self.num_key_value_groups = config['num_attention_heads'] // config['num_key_value_heads']
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj_bias = model_weights[f'model.layers.{layer_idx}.self_attn.q_proj.bias']
        self.q_proj_weight, self.q_proj_dequantize = get_weights(model_weights, f'model.layers.{layer_idx}.self_attn.q_proj.weight')
        self.k_proj_bias = model_weights[f'model.layers.{layer_idx}.self_attn.k_proj.bias']
        self.k_proj_weight, self.k_proj_dequantize = get_weights(model_weights, f'model.layers.{layer_idx}.self_attn.k_proj.weight')
        self.v_proj_bias = model_weights[f'model.layers.{layer_idx}.self_attn.v_proj.bias']
        self.v_proj_weight, self.v_proj_dequantize = get_weights(model_weights, f'model.layers.{layer_idx}.self_attn.v_proj.weight')
        self.o_proj_weight, self.o_proj_dequantize = get_weights(model_weights, f'model.layers.{layer_idx}.self_attn.o_proj.weight')

    def __call__(
        self,
        hidden_states: np.ndarray,
        position_embeddings: tuple,
        past_key_value: DynamicCache = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # 计算 Q、K、V，reshape、转置 是为了多头注意力
        query_states = (hidden_states @ dequantize(self.q_proj_weight, self.q_proj_dequantize) + self.q_proj_bias).reshape(hidden_shape).transpose((0, 2, 1, 3))
        key_states   = (hidden_states @ dequantize(self.k_proj_weight, self.k_proj_dequantize) + self.k_proj_bias).reshape(hidden_shape).transpose((0, 2, 1, 3))
        value_states = (hidden_states @ dequantize(self.v_proj_weight, self.v_proj_dequantize) + self.v_proj_bias).reshape(hidden_shape).transpose((0, 2, 1, 3))
        # 计算 Q、K 的 RoPE，位置编码
        cos, sin = position_embeddings
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states   = (key_states   * cos) + (rotate_half(key_states  ) * sin)
        # 加载 KV缓存
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        if self.num_key_value_groups > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        # 计算注意力分数: scores = Q @ K^T / sqrt(d_k)
        scores = np.einsum("...id,...jd->...ij", query_states, key_states) * self.scaling
        # 以下代码等效于上面的 np.einsum
        # key_dims = np.arange(key_states.ndim)
        # key_T = key_states.transpose(tuple(key_dims[:-2]) + (key_dims[-1], key_dims[-2]))
        # scores = query_states @ key_T * self.scaling
        if self.is_causal and query_states.shape[-2] != 1:
            attn_mask = np.tril(np.ones_like(scores, dtype=np.int32))
            scores[attn_mask == 0] = -float("inf") # MASK 未来信息
        # 计算注意力输出: attn_output = softmax(scores) @ V
        attn_output = np.einsum("...tij,...tjd->...itd", softmax(scores, axis=-1), value_states)
        # 以下代码等效于上面的 np.einsum
        # attn_output = softmax(scores, axis=-1) @ value
        # attn_output = attn_output.transpose((0, 2, 1, 3))

        # 多头注意力输出: attn_output = concat(attn_output) @ wO，上面的实现可并行计算 concat(attn_output) 部分，还需要一个矩阵乘法 wO
        attn_output = attn_output.reshape(*input_shape, -1)
        return attn_output @ dequantize(self.o_proj_weight, self.o_proj_dequantize)


class Qwen2MLP():
    """简单的 MLP 层，中间使用 SiLU 激活函数"""
    def __init__(self, model_weights: np.ndarray, layer_idx: int):
        self.gate_proj_weight, self.gate_proj_dequantize = get_weights(model_weights, f'model.layers.{layer_idx}.mlp.gate_proj.weight')
        self.up_proj_weight, self.up_proj_dequantize = get_weights(model_weights, f'model.layers.{layer_idx}.mlp.up_proj.weight')
        self.down_proj_weight, self.down_proj_dequantize = get_weights(model_weights, f'model.layers.{layer_idx}.mlp.down_proj.weight')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gate_proj = x @ dequantize(self.gate_proj_weight, self.gate_proj_dequantize)
        up_proj = x @ dequantize(self.up_proj_weight, self.up_proj_dequantize)
        down_proj = (silu(gate_proj) * up_proj) @ dequantize(self.down_proj_weight, self.down_proj_dequantize)
        return down_proj


class Qwen2DecoderLayer():
    def __init__(self, config: dict, model_weights: dict, layer_idx: int):
        self.hidden_size = config['hidden_size']
        self.self_attn = Qwen2Attention(config=config, model_weights=model_weights, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(model_weights=model_weights, layer_idx=layer_idx)
        self.input_layernorm = Qwen2RMSNorm(model_weights[f'model.layers.{layer_idx}.input_layernorm.weight'], eps=config['rms_norm_eps'])
        self.post_attention_layernorm = Qwen2RMSNorm(model_weights[f'model.layers.{layer_idx}.post_attention_layernorm.weight'], eps=config['rms_norm_eps'])

    def __call__(
        self,
        hidden_states: np.ndarray,
        past_key_value: DynamicCache = None,
        position_embeddings: tuple = None,  # necessary, but kept here for BC
    ):
        # 计算 Attention 和 残差连接
        hidden_states += self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        # 计算 MLP 和 残差连接
        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states


class Embedding:
    """嵌入层：将输入 ids 的 onehot编码 映射到隐藏层纬度(词嵌入纬度)，实际上是一个矩阵乘法"""
    def __init__(self, model_weights: dict):
        self.weights, self.dequantize = get_weights(model_weights, f'model.embed_tokens.weight')
        self.one_hot_encoder = np.arange(self.weights.shape[0])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (self.one_hot_encoder == x[..., None]).astype(self.weights.dtype) @ dequantize(self.weights, self.dequantize)


class Model:
    """汇总后简化的模型类，整理了所有的子模块"""
    def __init__(self, model_weights_path: str, configs_path: str = None, tokenizer_path: str = None):
        self.model_weights = load_model(model_weights_path)
        if configs_path is None:
            configs_path = os.path.join(model_weights_path, 'config.json')
        with open(configs_path, 'r') as configs_file:
            self.config = json.load(configs_file)
        if tokenizer_path is None:
            tokenizer_path = os.path.join(model_weights_path, 'tokenizer.json')
        self.tokenizer:Tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_token_id = [self.config["bos_token_id"], self.config["eos_token_id"]]
        self.pad_token_id = self.config["bos_token_id"]
        self.tokenizer.enable_padding(direction="left", pad_id=self.pad_token_id)
        self.embed_tokens = Embedding(self.model_weights)
        self.vocab_size = self.config['vocab_size']        # 词表大小
        self.decode_layers_num = self.config['num_hidden_layers']

        self.layers = [
            Qwen2DecoderLayer(self.config, self.model_weights, layer_idx) for layer_idx in range(self.decode_layers_num)
        ]

        self.norm = Qwen2RMSNorm(self.model_weights['model.norm.weight'], eps=self.config['rms_norm_eps'])
        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        if 'lm_head.weight' in self.model_weights.keys(): # 是否共享词嵌入层和输出层权重
            self.lm_head_weights, self.lm_head_dequantize = get_weights(self.model_weights, f'lm_head.weight')
        else:
            self.lm_head_weights = None # self.model_weights[f'model.embed_tokens.weight'].T

    def __call__(
        self,
        input_ids: np.ndarray,
        past_key_values: DynamicCache = None,
        num_logits_to_keep: int = 0,
    ) -> np.ndarray:
        # ids 映射到 隐藏层纬度：[批大小, 序列长度, hidden_size]
        hidden_states = self.embed_tokens(input_ids)
        
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = np.arange(past_seen_tokens, past_seen_tokens + hidden_states.shape[1])[None]
        # 共享位置编码
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # 逐层计算，尺寸始终不变：[批大小, 序列长度, hidden_size]
        for decoder_layer in self.layers[: self.decode_layers_num]:
            hidden_states = decoder_layer(
                hidden_states,
                past_key_value=past_key_values,
                position_embeddings=position_embeddings,
            )
        # 输出层：[批大小, 序列长度, vocab_size]
        hidden_states = self.norm(hidden_states)
        if self.lm_head_weights is None:
            lm_head_weights = dequantize(self.embed_tokens.weights, self.embed_tokens.dequantize).T
        else:
            lm_head_weights = dequantize(self.lm_head_weights, self.lm_head_dequantize)
        logits = hidden_states[:, -num_logits_to_keep:, :] @ lm_head_weights # 只保留最后 num_logits_to_keep(默认1) 个 token 的 logits
        # logits：[批大小, num_logits_to_keep(默认1), vocab_size]
        return logits

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 512) -> np.ndarray:
        """持续生成。
        结束条件1：生成的 token 数量达到 max_new_tokens。
        结束条件2：生成的 token 包含结束标记 eos_token_id。
        """
        batch_size = input_ids.shape[0]
        past_key_values = DynamicCache()
        # past_key_values = None
        unfinished_sequences = np.ones(batch_size, dtype=np.int32)
        eos_token_id_flag = np.ones(batch_size, dtype=np.bool)
        max_length = max_new_tokens + input_ids.shape[-1]
        this_peer_finished = False
        model_input_ids_len = input_ids.shape[1]
        is_benckmark = True
        if is_benckmark:
            start_time = 0
        while not this_peer_finished:
            outputs = model(
                # 第一次输入时，input_ids 为完整的对话文本，之后每次输入只包含最后一个 token，先前的 token 从 KV缓存 中获取
                input_ids=input_ids if past_key_values is None or past_key_values.get_seq_length()==0 else input_ids[:, -1:], 
                past_key_values=past_key_values,
                num_logits_to_keep=1,
            )
            if is_benckmark and start_time == 0:
                start_time = time.perf_counter()
            next_token_logits = outputs[:, -1, :]
            next_token_scores = logits_processor(input_ids, next_token_logits) # 后处理
            probs = softmax(next_token_scores) # 计算概率
            next_tokens = np.zeros(batch_size, dtype=np.int64)
            for batch_idx in range(batch_size):
                # 从概率分布中采样一个 token，作为输出
                next_tokens[batch_idx] = np.random.choice(np.arange(probs.shape[-1], dtype=np.int64), 1, p=probs[batch_idx])[0]
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
            input_ids = np.concat([input_ids, next_tokens[:, None]], axis=-1) # 将新 token 添加到输入中

            for batch_idx in range(batch_size):
                eos_token_id_flag[batch_idx] = int(input_ids[batch_idx, -1]) in self.eos_token_id
            stopping_criteria_flag = np.full(unfinished_sequences.shape, input_ids.shape[-1] >= max_length, np.bool)
            
            unfinished_sequences = unfinished_sequences & ~(stopping_criteria_flag|eos_token_id_flag)
            this_peer_finished = unfinished_sequences.max() == 0

            if is_benckmark:
                continue # 跳过打印输出
            generated_ids = [
                output_ids[model_input_ids_len:] for output_ids in input_ids
            ]
            # 实时解码输出，打印生成的 token
            response = self.tokenizer.decode_batch(generated_ids, skip_special_tokens=True)
            print(unfinished_sequences, this_peer_finished, f'{"\n".join(response)}')
        if is_benckmark:
            end_time = time.perf_counter()
            token_count = input_ids.shape[-1] - model_input_ids_len - 1
            print(f"Tokens per second: {token_count * batch_size / (end_time - start_time):.2f}")
        generated_ids = [
            output_ids[model_input_ids_len:] for output_ids in input_ids
        ]
        return generated_ids


if __name__ == '__main__':
    # chat_template = '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
    # model_weights_path = '/Users/acai/Downloads/models/Qwen2.5_0.5B_Instruct_npy_FP16'
    chat_template = '<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜><think>\n'
    model_weights_path = '/Users/acai/Downloads/models/DeepSeek_R1_Distill_Qwen_1.5B_npy_FP32'

    model = Model(model_weights_path)

    role_system_content = "You are a helpful assistant."
    prompt = [
        # "怎么用python numpy实softmax？",
        "你是谁？",
        # "计算456+826",
    ] # 批次
    text = list(map(lambda x: chat_template.format(role_system_content, x), prompt))

    model_inputs = np.array([model.tokenizer.encode_batch_fast(text)[i].ids for i in range(len(text))], dtype=np.int32)

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=2048
    )

    response = model.tokenizer.decode_batch(generated_ids, skip_special_tokens=True)
    print(f'{"\n".join(response)}')
