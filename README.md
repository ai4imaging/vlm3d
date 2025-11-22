# VLM3D: Let Language Constrain Geometry 
## Vision-Language Models as Semantic and Spatial Critics for 3D Generation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.14271-b31b1b.svg)](https://arxiv.org/abs/2511.14271)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ai4scientificimaging.org/vlm3d/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

</div>

**Official Implementation of "Let Language Constrain Geometry: Vision-Language Models as Semantic and Spatial Critics for 3D Generation"**

VLM3D is a general framework that repurposes large vision-language models (specifically Qwen2.5-VL) as powerful, differentiable semantic and spatial critics for 3D generation.

---

## 🛠️ Installation

We recommend using Anaconda to manage the environment.

### 1. Clone the repository
```bash
git clone [https://github.com/ai4imaging/vlm3d.git](https://github.com/ai4imaging/vlm3d.git)
cd vlm3d
```

### 2. Conda environment
```bash
conda create -n vlm3d python=3.10
conda activate vlm3d

# Install PyTorch (adjust cuda version according to your system)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# Install required packages
pip install -r requirements.txt
```


### 3. ⚠️ Important: Patching Transformers for Differentiability
Critically Important: Standard VLM image processors detach gradients, which breaks the differentiable reward mechanism required for VLM3D.

You MUST replace the official image_processing_qwen2_vl.py in your installed transformers library with the one provided in this repository.

```bash
# Locate your transformers installation path
TRANSFORMERS_PATH=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")

# Replace the file (Assuming our modified version is in ./custom/ or root)
# Please check the 'custom' folder or root for the modified file
cp ./custom/image_processing_qwen2_vl.py $TRANSFORMERS_PATH/models/qwen2_vl/image_processing_qwen2_vl.py

echo "Patch applied successfully to: $TRANSFORMERS_PATH"
```
This modification ensures that image processing operations use Torch tensors exclusively, maintaining an uninterrupted gradient flow for the VLM critic.


### 4. 🚀 Quick Start
To generate a 3D asset using the default configuration (as shown in the paper):

```bash
sh generate.sh
```



### 📝 Citation
If you find this code or paper useful for your research, please cite:

```bibtex
@article{bai2026vlm3d,
  title={Let Language Constrain Geometry: Vision-Language Models as Semantic and Spatial Critics for 3D Generation},
  author={Bai, Weimin and Li, Yubo and Luo, Weijian and Lai, Zeqiang and Wang, Yequan and Chen, Wenzheng and Sun, He},
  journal={ArXiv},
  year={2025}
}
```