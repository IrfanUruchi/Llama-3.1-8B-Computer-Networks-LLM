# Llama-3.1-8B-Computer-Networks-LLM

**A fine-tuned 8B LLaMA model specialized for computer network-related tasks.**

> **Note:** This repository and model are provided for research purposes only. Use of the underlying LLaMA model is subject to Meta AI's [LLaMA License](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/). By using this repository, you agree to comply with all applicable terms and conditions of the LLaMA license.


---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Card](#model-card)
- [Compliance and Licensing](#compliance-and-licensing)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Llama-3.1-8B-Computer-Network-LLM** is a fine-tuned version of Meta's LLaMA-3.1-8B model. The fine-tuning process was performed on computer network-related text, making it suitable for generating answers and explanations in the networking domain.

In the repository are included:
Configuration files to run inference and also a set of links to download the large safetensors model shards hosted on MEGA. 

---

## Installation

### Prerequisites

- **Python 3.12 or 3.13**
- **Git** for pulling the repository files (or you can download the configuration files one by one)
- **PyTorch** and the [Transformers](https://huggingface.co/transformers/) library

- ### Code for cloning the repository (code is for bash linux/macOS)

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/Llama-3.1-8B-Computer-Network-LLM.git
cd Llama-3.1-8B-Computer-Network-LLM
```

### Dependecies and Enviroment (optional but recommended)

```bash
python -m venv myenv
source myenv/bin/activate
pip install torch transformers datasets peft huggingface_hub
```

### Model files

The large safetensor model shards are not stored in this repository. Instead i have hosted them in MEGA , there are 6 files totalling around 11GB :

- [model-00001-of-00006.safetensors:](https://mega.nz/file/rppWmDpS#X5utsf27-npdkFQVCQzz_gFi-s5a4oCuUSUYtJDw6p4)
  
- [model-00002-of-00006.safetensors:](https://mega.nz/file/jkRDVapZ#QhG5Pl8mu-DORIqCvaOfEcHspcVV79Xu-nxiaSa8pmA)
  
- [model-00003-of-00006.safetensors:](https://mega.nz/file/fsQBjQ6D#MI9gi1L9BDycxGh8qE9D92Q1IiJIMkujFwGeel60rk0)
  
- [model-00004-of-00006.safetensors:](https://mega.nz/file/7lB3GQZT#va8qP_X-ADHwtmgyxNcGhRklZ6TKFMg9JuNT7Xbl0js)

- [model-00005-of-00006.safetensors:](https://mega.nz/file/n0oS2IoQ#toljZ9fC2pG1r7WTHO_rHhBYC1qv2lGI6Jg_UgwKWS8)
  
- [model-00006-of-00006.safetensors:](https://mega.nz/file/2xQQBbKL#QMpL6l8bymBtAJnJPzZibcd8U3vv9b4BeQY7D4vcr0U)

After downloading , place all the safetensors files into the folder with the other configuration file in your local copy of the repository. **Ensure that the model loading scripts point to the correct directory**.


## Usage

Below is a example Python script to load the model (assuming the model files have been placed in a folder called **model_files**) and run inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_dir = os.path.join(os.getcwd(), "model_files")


model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# You may need to change if you have diffrent GPU brand (for the CUDA part else it will direct to CPU either way) like if you have Apple M series of chips, AMD or Intel 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Example prompt for a computer network-related question
prompt = "Explain simply how OSPF and BGP work with TCP/IP to route data on the Internet."

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=250, do_sample=True, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response:")
print(generated_text)
```

# Model card 

- ** Model Name:** Llama-3.1-8B-Computer-Network-LLM
- ** Base Model: ** LLaMA-3.1-8B (Meta AI)
- **Domain:** Computer Networks

 ## ** Intended use:
 ** Research and academic study and for generating network-related explanations and answers.


## Limitations:
It's not for commercial use and may produce biased or inaccurate results , use it with caution and adjust prompts as needed.


# Compliance and Licensing

This project uses Meta AI's LLaMA-3.1-8B model under the terms of the LLaMA License:
https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE

By using this model, you agree to: 
Use the model for reseach and non-commercial purposes only, provide proper attribution to Meta AI when using or referencing the model, abide by any restrictions specified in the LLaMA license.

And you can review the full licence for more details.

# Acknowledgements

Meta AI for providing the LLaMA model.

Hugging Face for the Transformers library and tools.

This project is maintained for research purposes and is not intended for commercial deployment.
