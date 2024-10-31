
# Mixture of Adversarial LoRAs: Boosting Robust Generalization in Meta-tuning

This is the code repository for paper [Mixture of Adversarial LoRAs: Boosting Robust Generalization in Meta-tuning](https://openreview.net/forum?id=HxGdbAmYYr), NIPS 2024.

###  Abstract
This paper introduces AMT, an \textbf{A}dversarial \textbf{M}eta-\textbf{T}uning methodology, to boost the robust generalization of pre-trained models in the out-of-domain (OOD) few-shot learning. To address the challenge of transferring knowledge from source domains to unseen target domains, we construct the robust LoRAPool by meta-tuning LoRAs with dual perturbations applied to not only the inputs but also singular values and vectors of the weight matrices at various robustness levels. On top of that, we introduce a simple yet effective test-time merging mechanism to dynamically merge discriminative LoRAs for test-time task customization. Extensive evaluations demonstrate that AMT yields significant improvements, up to 12.92\% in clean generalization and up to 49.72\% in adversarial generalization, over previous state-of-the-art methods across a diverse range of OOD few-shot image classification tasks on three benchmarks, confirming the effectiveness of our approach to boost the robust generalization of pre-trained models.



### Environment Preparation
```
pip install -r requirements.txt
```
The code was tested with Python 3.9.0 and Pytorch >= 1.10.0.


### Data Preparation
- Meta-Dataset: ImageNet, Omniglot, Aircraft, CUB, DTD, QuickDraw, Fungi, VGG Flower,
Traffic Signs, MSCOCO
- BSCD-FSL benchmark: CropDisease, EuroSAT, ISIC, ChestX
- Fine-grained datasets: CUB, Car, Plantae, Places



### Training
We use official pretrained foundation models, e.g., `DINO ViT-small`, `iBOT ViT-small`, `DeIT ViT-small`.
```
python main.py  --dataset meta_dataset --data-path path_to_data_files --base_sources ilsvrc_2012  --arch dino_small_patch16_amt --output experiment_name  --fp16 --device cuda:0
```

### Test
``` 
python test_meta_dataset.py  --dataset meta_dataset --data-path path_to_data_files --arch dino_small_patch16_amt_merge --deploy merge-vanilla --output experiment_name --resume checkpoint.pth --device cuda:0 --test_sources target_domain_name
``` 

Below are the meta-tuned weights for Table 1/2/3:

Setting |  Weights            |
---|---|
5-way 1-shot | [checkpoint](./checkpoints/metadataset_imagenet_advmetatuned_5w1s.pth)
5-way 5-shot | [checkpoint](./checkpoints/metadataset_imagenet_advmetatuned_5w5s.pth)

------
If you find our project helpful, please consider cite our paper:
```
@inproceedings{advmixture24,
title={Mixture of Adversarial Lo{RA}s: Boosting Robust Generalization in Meta-tuning},
author={Yang, Xu and Liu, Chen and Wei, Ying},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```