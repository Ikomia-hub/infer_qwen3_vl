<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_qwen3_vl</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_qwen3_vl">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_qwen3_vl">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_qwen3_vl/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_qwen3_vl.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) is the multimodal large language model series developed by Qwen team, Alibaba Cloud.

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow
```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_qwen3_vl", auto_connect=True)

# Run on your image  
wf.run_on(url='https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_text_inspiration.jpg?raw=true')

# Save output .json
qwen_output = algo.get_output(1)
qwen_output.save('qwen_output.json')
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
| Parameters             | Description |
|----------------------|-------------|
| `input_path`       | Path to a single PDF file to process or to a directory containing multiple PDFs. |
| `model_name`       | Name or path of the Qwen VL model. Default: `"Qwen/Qwen3-VL-4B-Instruct"`. |
| `prompt`          | Custom prompt to guide the model's response for the given image. Default: `"Read all the text in the image."` |
| `system_prompt`   | System prompt to set the behavior and context for the model. Default: `"You are a helpful assistant."` |
| `cuda`             | If True, CUDA-based inference (GPU). If False, run on CPU. |
| `do_sample`       | Whether or not to use sampling ; use greedy decoding otherwise (return the word/token which has the highest probability). If set to `True`, token validation incorporates resampling for generating more diverse outputs. Acceptable values are `True` or `False`. Default: `False`. |
| `max_new_tokens`  | The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. Default: `1280`. *(For `essais` reports, reducing this value can significantly speed up inference time. Lower values are recommended for `essais` to mitigate hallucinations.)* |
| `temperature`     | Sampling temperature for text generation. Default: `1`. *(Only used if `--do_sample=True`.)* |
| `top_p`           | Top-p sampling parameter. Default: `1`. *(Only used if `--do_sample=True`.)* |
| `top_k`           | Top-k sampling parameter. Default: `50`. *(Only used if `--do_sample=True`.)* |
| `repetition_penalty` | The parameter for repetition penalty. 1.0 means no penalty. . Default: `1.0`.|


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_qwen3_vl", auto_connect=True)

algo.set_parameters({
    "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
    "cuda": "True",
    "prompt": "Describe the image in detail.",
    "max_new_tokens": "512", 
    "do_sample": "False",
    "temperature": "1",
    "top_p": "1",
    "top_k": "50",
    "repetition_penalty": "1.0"
})

# Run on your image  
wf.run_on(url='https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_text_inspiration.jpg?raw=true')

# Save output .json
qwen_output = algo.get_output(1)
qwen_output.save('qwen_output.json')
```

## :mag: Explore algorithm outputs
Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_qwen3_vl", auto_connect=True)

# Run on your image  
wf.run_on(url='https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_people_workspace.jpg?raw=true')

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
