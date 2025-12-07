# Agent-based traffic model
Agent-based traffic simulation at an intersection using parallel computing.

# Prerequisites
1. In order to run CUDA acceleration you have to install latest NVidia drivers and CUDA Toolkit.

* https://www.nvidia.com/en-us/drivers/
* https://developer.nvidia.com/cuda-toolkit

2. Ensure you have either **Anaconda** or **Miniconda** installed on your system.

**Miniconda** is recommended for its smaller size and quicker installation. You can download it from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

### Install required libraries and create conda env

```bash
conda create -n traffic-sim python=3.11
conda activate traffic-sim
pip install -r requirements.txt
```