# 阿里通义千问 Qwen2.5 numpy推理(支持Deekseek-R1蒸馏的Qwen模型)

- 只使用 `numpy` 实现 `Qwen2.5` 的推理，不使用 `torch`、`transformers` 等框架，易于学习LLM的推理过程，以及移植到其它语言
- 支持阿里云原始的通义千问 `Qwen2.5` 模型、`Deekseek-R1` 蒸馏的 `Qwen2.5` 模型，其它微调模型暂未测试(理论上支持)
- 支持 `batch` 推理
- 支持 `temperature`、`topk`、`topp`、`penalty`等参数
- 支持 `KV缓存`
- 支持 `q8_0量化`
- 以学习为目的，约400行代码实现了完整的llm推理过程，不含 `tokenization` 部分

## 测试

### 1. 安装依赖

```bash
pip install numpy tokenizers
```

### 2. 下载 `safetensors` 模型

到模型分享平台下载完整模型，参考 [`modelscope` 平台下载说明](https://www.modelscope.cn/docs/models/download)

  1. [Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary)
  2. [Qwen2.5-1.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/summary)
  3. [DeepSeek-R1-Distill-Qwen-1.5B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/summary)

### 3. 转换模型

使用 `parse_safetensors.py` 脚本转换模型，提供下载的模型目录，转换后的npy模型保存目录，例如:

```bash
python parse_safetensors.py --model_dir 下载的模型目录 --npy_save_dir 转换后的npy模型保存目录
```

### 4. 运行推理

修改 `model.py` 中的模型路径、`prompt`，运行 `model.py`，或者手动从 `model.py` 中导入 `Model` 类，参考 `model.py` 中 `main` 函数的使用方法

```python
if __name__ == '__main__':
    # chat_template = '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
    # model_weights_path = '/Users/acai/Downloads/models/Qwen2.5_1.5B_Instruct_npy'
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
```

## Benchmark

与 [`llama.cpp`](https://github.com/ggml-org/llama.cpp/releases/tag/b4722)对比每秒 `Tokens` 速度，测试平台为 `Mac mini M4`，内存16G

|模型|精度|numpy|llama.cpp|
|:---:|:---:|:---:|:---:|
|Qwen2.5_0.5B_Instruct|float32|29.77|45.6|
|Qwen2.5_0.5B_Instruct|float16|-|86.44|
|Qwen2.5_0.5B_Instruct|q8_0|1.94|140.53|
|DeepSeek_R1_Distill_Qwen_1.5B|float32|10.31|15.55|
|DeepSeek_R1_Distill_Qwen_1.5B|float16|-|31.55|
|DeepSeek_R1_Distill_Qwen_1.5B|q8_0|0.68|54.47|

7B模型用float32精度时需要30G左右内存，机器内存不足，未测试

比较震惊的是 `numpy` 的矩阵加速只支持float32、float64，不支持整数、半精度等，导致float32速度是最快的，float32模型内存占用很大，容易导致内存不足，同时对内存带宽的要求很高，估计只能通过移植到其它语言，从底层优化矩阵运算才能加速到 `llama.cpp` 的速度

## 参考

- [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [transformers](https://github.com/huggingface/transformers)
