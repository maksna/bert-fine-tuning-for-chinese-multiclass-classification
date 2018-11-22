#bert-chinese classification

Use google BERT to do chinese sentences multiclass classification !

#train

export BERT_BASE_DIR=/path/to/model/chinese_L-12_H-768_A-12

export DATA_DIR=/path/to/data

python run_chinese_classification.py \
  --do_train=true \
  --do_eval=true \
  --train_dir=$DATA_DIR/Chinesedata/train.tsv \
  --dev_dir=$DATA_DIR/Chinesedata/dev.tsv \
  --test_dir=$DATA_DIR/Chinesedata/test.tsv \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=6 \
  --learning_rate=1e-6 \
  --num_train_epochs=10.0 \
  --output_dir=/path/to/chinese_model

#test

export BERT_BASE_DIR=/path/to/model/chinese_L-12_H-768_A-12

export DATA_DIR=/path/to/model/data

export TRAINED_CLASSIFIER=/path/to/model/chinese_model


python run_chinese_classification.py \
  --train_dir=$DATA_DIR/Chinesedata/train.tsv \
  --dev_dir=$DATA_DIR/Chinesedata/dev.tsv \
  --test_dir=$DATA_DIR/Chinesedata/test.tsv \
  --do_predict=true \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=512 \
  --output_dir=/path/to/chinese_result
 
#results:

python results.py

#data

url：https://pan.baidu.com/s/1qDngiTq1FyNxb5GX-zDiCg 
passwd：9dk8 

#reference:

https://github.com/google-research/bert

https://arxiv.org/abs/1810.04805









