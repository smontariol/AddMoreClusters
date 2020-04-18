# AddMoreClusters #
Shared repository with code from paper "Capturing Evolution in Word Usage: Just Add More Clusters?" presented at WWW2020 Temporal Web workshop.

## Installation, documentation ##

Published results were produced in Python 3 programming environment. Instructions for installation assume the usage of PyPI package manager.<br/>
To get the source code, clone the project from the repository with 'git clone https://github.com/smontariol/AddMoreClusters/'<br/>
Install dependencies if needed: pip install -r requirements.txt

### To reproduce the results on the COHA corpus published in the paper run the code in the command line using following commands: ###

To fine-tune custom BERT model on the COHA corpus<br/>
```bash
python3 BERT_finetuning.py --train_data_file PATH_TO_COHA_CORPUS --output_dir PATH_TO_SAVED_MODEL --model_type bert --mlm --do_train --num_train_epochs 5 --per_gpu_train_batch_size 8 --model_name_or_path bert-base-uncased
```
To generate BERT contextual embeddings embeddings<br/>
TODO

To cluster contextual embeddings embeddings<br/>
TODO

To detect semantic drift for each target word<br/>
TODO

## Ackowledgements
This work has been partly supported by the European Union’s Horizon 2020 research and innovation programme under grant 770299 (NewsEye) and 825153 (EMBEDDIA).


