# GDAS-PSL: Generative and Dual-Attention Suppression Models for Complex Protein Subcellular Localization



This is the official PyTorch code for the paper:

**GDAS-PSL: Generative and Dual-Attention Suppression Models for Complex Protein Subcellular Localization（修订版）**

**Sina Xia**, **Xiaoli Ruan**,**Jing Yang**,**Quan Zhou**,**Guangfeng Lin**, **Suhao Chen**, **Yajun Chen**

**Paper**|**Code**

![Uploading 方法的图.jpg…](https://github.com/ruanxiaoli/GDAS-PSL/blob/main/Method.jpg)


## Setup



The model code is implemented based on the PyTorch framework. The experimental environment includes:

- NVIDIA A40 GPU

Create a conda environment `py38` using:

```
	conda env create -f environment.yml
	conda activate py38
```



## Preparation



The datasets utilized in our work (D1, D2, D3) must be prepared prior to model training and testing. The data is downloaded from The Human Protein Atlas (HPA) database（The Human Protein Atlas](https://www.proteinatlas.org/)）） and can be obtained using the following code:

```
	python utils/down_img.py
```



## Training



If you need to train the diffusion model in the paper and process low-quality datasets, the command is as follows:

```
    python diffusionModel/main.py --config diffusionModel/configs/config-rawUnet.yaml
```



To train a classifier for judging the sampling step, the command is as follows:

```
	python diffusionModel/classifier.py
```



The command to train the model using our **GDAS-PSL** is as follows:

```
    python TrainD1_block4_m95_r7.py
```



## Test



Sampling low-quality data based on the diffusion model, the command is as follows:

```
	python diffusionModel/inference_recalibrate.py --config configs/config-inference-recalibrate_raw.yaml
```



Perform low-quality data enhancement based on the diffusion model. The command is as follows:

```
	python diffusionModel/diffusion_enhanced_img.py
```





Testing the **GDAS-PSL** model:

```
	python TestD1.py
```



## References



If you use this code in your research, please consider citing our paper:

```
@article{,
  title={GDAS-PSL: Generative and Dual-Attention Suppression Models for
Complex Protein Subcellular Localization},
  author={Sina Xia, Xiaoli Ruan, Jing Yang, Quan Zhou, Guangfeng Lin, Suhao Chen, Yajun Chen},
  booktitle={},
  year={},
  organization={}
}

```



**If you have any questions you can contact us**:gs.snxia23@gzu.edu.cn **or**xlruan@gzu.edu.cn.
