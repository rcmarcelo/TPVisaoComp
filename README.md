# Detecção Multi-nível de Tecidos Malignos em Colonoscopias utilizando GANs

## Ambiente para execução
`conda`: 
```
$ conda env create -f digest_env.yml
$ conda activate digest_env
```

`apex` : Caso seja necessário reinstalar
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Conjunto de dados
Organização:
```
├── data/
│   ├── tissue-train-neg/     
│   ├── tissue-train-pos-v1/
```
[Download](https://drive.google.com/drive/folders/1_19Nz7mPuLReYA60UAtcnsAotTqZk0Je)

## Pré-processamento
```
$ cd code/
$ python preprocessing.py
```

## Treinamento
```
$ cd code/
$ python train.py --config_file='config/cac-unet-r50.yaml'
```
## Créditos
ZHU, Chuang et al. Multi-level colonoscopy malignant tissue detection with adversarial CAC-UNet. Neurocomputing, v. 438, p. 165-183, 2021.
