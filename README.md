# **IDOL: Instant Photorealistic 3D Human Creation from a Single Image**  

[![Website](https://img.shields.io/badge/Project-Website-0073e6)](https://yiyuzhuang.github.io/IDOL/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2412.14963)
[![Live Demo](https://img.shields.io/badge/Live-Demo-34C759)](https://yiyuzhuang.github.io/IDOL/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


---

<p align="center">
  <img src="./asset/images/Teaser_v2.png" alt="Teaser Image for IDOL" width="85%">
</p>

---

## **Abstract**

This work introduces **IDOL**, a feed-forward, single-image human reconstruction framework that is fast, high-fidelity, and generalizable. 
Leveraging a large-scale dataset of 100K multi-view subjects, our method demonstrates exceptional generalizability and robustness in handling diverse human shapes, cross-domain data, severe viewpoints, and occlusions. 
With a uniform structured representation, the reconstructed avatars are directly animatable and easily editable, providing a significant step forward for various applications in graphics, vision, and beyond.

In summary, this project introduces:

- **IDOL**: A scalable pipeline for instant photorealistic 3D human reconstruction using a simple yet efficient feed-forward model.
- **HuGe100K Dataset**: We develop a data generation pipeline and present \datasetname, a large-scale multi-view human dataset featuring diverse attributes, high-fidelity, high-resolution appearances, and a well-aligned SMPL-X model.
- **Application Support**: Enabling 3D human reconstruction and downstream tasks such as editing and animation.


---
## 📰 **News** 

- **2024-12-18**: Paper is now available on arXiv.
- **2025-01-02**: The demo dataset containing 100 samples is now available for access. The remaining dataset is currently undergoing further cleaning and review.
- **2025-03-01**: 🎉 Paper accepted by CVPR 2025.
- **2025-03-01**:  🎉 We have released the inference code! Check out the [Code Release](#code-release) section for details.


## 🚧 **Project Status**   

We are actively working on releasing the following resources:  

| Resource                    | Status              | Expected Release Date      |
|-----------------------------|---------------------|----------------------------|
| **Dataset Demo**            | ✅ Available        | **Now Live! (2025.01.02)**      |
| **Inference Code**             | ✅ Available        | **Now Live! (2025.03.01)**   |
| **Online Demo**             | 🚧 In Progress      | **Before April 2025**   |
| **Training Code**                    | 🚧 In Progress      | **Before April 2025**   |
| **Full Dataset Access**          | 🚧 Coming Soon      | **Before April 2025**   |

Stay tuned as we update this section with new releases! 🚀  



## 💻 **Code Release** 

### Installation & Environment Setup

Please refer to [env/README.md](env/README.md) for detailed environment setup instructions.

### Quick Start
Run demo with different modes:
```bash
# Reconstruct the input image
python run_demo.py --render_mode reconstruct

# Generate novel poses (animation)
python run_demo.py --render_mode novel_pose

# Generate 360-degree view
python run_demo.py --render_mode novel_pose_A
```

<!-- 
### Training

```bash
python train.py --config configs/default.yaml
``` -->



## 🌐 **Key Links** 

- 📄 [**Paper on arXiv**](https://arxiv.org/pdf/2412.02684)  
- 🌐 [**Project Website**](https://yiyuzhuang.github.io/IDOL/)  
- 🚀 [**Live Demo**](https://your-live-demo-link.com) (Coming Soon!)  

---

## 📊 **Dataset Demo Access**   

We introduce **HuGe100K**, a large-scale multi-view human dataset, supporting 3D human reconstruction and animation research.  

### ▶ **Watch the Demo Video**
<p align="center">
  <img src="./asset/videos/dataset.gif" alt="Dataset GIF" width="85%">
</p>


### 🚀 **Access the Dataset**   

To access the **HuGe100K Dataset**, please submit your application using the link below:  

👉 [**Dataset Access Application Form**](https://docs.google.com/forms/d/e/1FAIpQLSeVqrA9Mc_ODdcTZsB3GgrxgSNZk5deOzK4f64N72xlQFhvzQ/viewform?usp=dialog)  

---

## 📝 **Citation**   

If you find our work helpful, please cite us using the following BibTeX:

```bibtex
@article{zhuang2024idolinstant,                
  title={IDOL: Instant Photorealistic 3D Human Creation from a Single Image}, 
  author={Yiyu Zhuang and Jiaxi Lv and Hao Wen and Qing Shuai and Ailing Zeng and Hao Zhu and Shifeng Chen and Yujiu Yang and Xun Cao and Wei Liu},
  journal={arXiv preprint arXiv:2412.14963},
  year={2024},
  url={https://arxiv.org/abs/2412.14963}, 
}
```



## **License** 

This project is licensed under the **MIT License**.

- **Permissions**: This license grants permission to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
- **Condition**: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- **Disclaimer**: The software is provided "as is", without warranty of any kind.

For more information, see the full license [here](https://opensource.org/licenses/MIT).

## **Support Our Work** ⭐

If you find our work useful for your research or applications:

- Please ⭐ **star our repository** to help us reach more people
- Consider **citing our paper** in your publications (see [Citation](#citation) section)
- Share our project with others who might benefit from it

Your support helps us continue developing open-source research projects like this one!

## 📚 **Acknowledgments**

This project is majorly built upon several excellent open-source projects:

- [E3Gen](https://github.com/olivia23333/E3Gen): Efficient, Expressive and Editable Avatars Generation
- [SAPIENS](https://github.com/facebookresearch/sapiens): High-resolution visual models for human-centric tasks
- [GeoLRM](https://github.com/alibaba-yuanjing-aigclab/GeoLRM): Large Reconstruction Model for High-Quality 3D Generation
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): Real-Time 3DGS Rendering

We thank all the authors for their contributions to the open-source community.
