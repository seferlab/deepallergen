# DeepAllergen: An Attention-based Approach Using Pretrained Models to Predict Allergens


## Installation
1. Download DeepAlgPro
```
git clone https://github.com/seferlab/deepallergen.git 
```
2. Install required packages<br>
```
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## Train and Test the model
```
usage: python train.py [-h] [-i INPUTS] [--epochs N] [--lr LR] [-b N] [--mode {train,test}]
```
### Optional arguments
```
  -h, --help            show this help message and exit
  -i INPUTS, --inputs INPUTS
  --epochs N            number of total epochs to run
  --lr LR, --learning-rate LR
                        learning rate
  -b N, --batch-size N
  --mode {train,test}
```
### Example
```
python train.py -i data/all.train.fasta --epochs 120 --lr 0.0001 -b 72 --mode train
```
## Use DeepAlgPro to predict allergens
```
usage: python predict.py [-h] [-i INPUTS] [-b N] [-o OUTPUT]
```
### Optional arguments
```
  -h, --help            show this help message and exit
  -i INPUTS, --inputs INPUTS
                        input file
  -b N, --batch-size N
  -o OUTPUT, --output OUTPUT
                        output file
```
### Example
```
python predict.py -i data/all.test.fasta -o allergen.predict.txt
```
