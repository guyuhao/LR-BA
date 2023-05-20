# LR-BA: Backdoor attack against vertical federated learning using local latent representations
This repository contains code for our paper: "LR-BA: Backdoor attack against vertical federated learning using local latent representations".
***
## Code usage: 
### Prepare dataset
1. Please create a **"data"** folder in the same directory of the code to save the raw dataset.
2. For MNIST, CIFAR-10 and CIFAR-100, torchvision is used to automatically download the raw dataset to **"data"** when network is available.
3. For CINIC-10, please download the raw dataset from https://github.com/BayesWatch/cinic-10 in advance and save and unzip it to **"data/CINIC"**. 
   Then, please create a new folder **"data/CINIC-L"**. Finally, execute the scripts provided in the dictionary **"data_preprocess"**.
   ```
   python3 cinic_enlarge.py
   python3 cinic_pickle.py
   ```
   The images for our code are saved in **"data/CINIC-L"**
4. For BHI, please download the raw dataset from https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images in advance and save and unzip it to **"data/BHI"**.
   Then, please execute the scripts provided in the dictionary **"data_preprocess"**.
   ```
   python3 bhi_preprocess.py
   python3 bhi_pickle.py
   ```
5. For Yahoo!, please download the raw dataset from https://www.kaggle.com/datasets/soumikrakshit/yahoo-answers-dataset in advance and save and unzip it to **"data/yahoo"**.
   Then, please execute the scripts provided in the dictionary **"data_preprocess"**.
   ```
   python3 yahoo.py
   ```
### Prepare configuration file
We provide a configuration file template **"config_template.yaml"** in the directory **"tests"** for running our code. You can refer to it to generate your own configuration file as needed. 

We also provide the configuration files of all experiments in our paper in the **"tests/final_config"** directory for reference.
### Run scripts
All experimental scripts are located in the dictionary **"tests"**. You can execute the scripts by providing the parameter **--config** which specified the configuration file path as follows.
1. To evaluate LR-BA
```angular2html
python3 vfl_backdoor_compare.py --config final_config/vfl_backdoor_compare/cifar10.yaml
```
2. To evaluate the impact of backdoor latent representation initialization
```angular2html
python3 initialization_compare.py --config final_config/vfl_backdoor_compare/cifar10.yaml
```
3. To evaluate the impact of optimization epochs setting in LR-BA
```angular2html
python3 generate_epochs_compare.py --config final_config/other/generate_epochs_compare_cifar10.yaml
```
4. To evaluate the impact of auxiliary labeled data size
```angular2html
python3 label_size_compare.py --config final_config/other/label_size_compare_cifar10.yaml
```
5. To evaluate the impact of unbalanced auxiliary data
```angular2html
python3 label_non_iid_compare.py --config final_config/other/label_non_iid_compare_cifar10.yaml
```
6. To evaluate the impact of benign passive parties
```angular2html
python3 party_compare.py --config final_config/other/local_mia/party_compare_bhi.yaml
```

Evaluate the effectiveness of LR-BA under defense.
1. To evaluate LR-BA under norm clipping defense
```angular2html
python3 defense_norm_clip.py --config final_config/defense/norm_clip_cifar10.yaml
```
2. To evaluate LR-BA under noisy gradients defense
```angular2html
python3 defense_noisy_gradient.py --config final_config/defense/noisy_gradient_cifar10.yaml
```
3. To evaluate LR-BA under gradient compression defense
```angular2html
python3 defense_gradient_compression.py --config final_config/defense/gradient_compression_cifar10.yaml
```
## Code architecture
```angular2html
.
├── common               # implementation of configuration parser
├── data_preprocess
├── datasets
├── model                # architecture of all target models and attack models
├── tests                # experimental scripts
│   ├── checkpoints      # to save model parameters during training
│   ├── data             # to save indices of training and testing dataset divided for each participant and for shadow model
│   ├── final_config     # to save configuration files of all experiments in our paper
│   └── result           # to save the log output by experimental scripts
└── vfl                  # implementation of vertical federated learning
    ├── backdoor         # implementation of LR-BA (lr_ba_backdoor.py) and other backdoor attacks
    └── defense          # implementation of three defenses: norm clipping, noisy gradients, and gradient compression
```

***
## Citation
If you use this code, please cite the following paper: 
### <a href="https://www.sciencedirect.com/science/article/pii/S0167404823001037">LR-BA</a>
```
@article{GU2023103193,
title = {LR-BA: Backdoor attack against vertical federated learning using local latent representations},
journal = {Computers & Security},
volume = {129},
pages = {103193},
year = {2023},
issn = {0167-4048},
doi = {https://doi.org/10.1016/j.cose.2023.103193},
url = {https://www.sciencedirect.com/science/article/pii/S0167404823001037},
author = {Yuhao Gu and Yuebin Bai},
keywords = {Vertical federated learning, Backdoor attack, Backdoor defense, Federated learning security, Artificial intelligence security},
abstract = {In vertical federated learning (VFL), multiple participants can collaborate in training a model with distributed data features and labels managed by one of them. The cooperation provides opportunities for a malicious participant to conduct a backdoor attack. However, the attack is challenging when the adversary does not own labels with the mitigation of other participants. In this paper, we discover that an adversary can exploit local latent representations output in the inference stage to inject a backdoor in VFL, even without access to labels. With little auxiliary labeled data, the adversary fine-tunes its bottom model to make it output specific latent representation for backdoor input instances, which induces the federated model to predict the attacker-specified label regardless of benign participants. Our experiments show that the proposed attack can achieve a high attack success rate with little loss of main task accuracy and outperform existing backdoor attacks. We also explore possible defenses against the attack. Our research demonstrates the potential security threat to VFL.}
}
```
