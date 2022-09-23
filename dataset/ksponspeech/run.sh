# Author
# Soohwan Kim, Seyoung Bae, Cheolhwang Won, Soyoung Cho, Jeongwon Kwak

DATASET_PATH="SET_YOUR_DATASET_PATH"
VOCAB_DEST='SET_LABELS_DESTINATION'
OUTPUT_UNIT='grapheme'                                          # you can set character / subword / grapheme
PREPROCESS_MODE='spelling'                                       # phonetic : 칠 십 퍼센트,  spelling : 70%
VOCAB_SIZE=5000                                                  # if you use subword output unit, set vocab size

# if you want to use pretrain kober tokenizer refer https://github.com/SKTBrain/KoBERT
# And release the bottom annotation.

python main.py \
--dataset_path $DATASET_PATH \
--vocab_dest $VOCAB_DEST \
--output_unit $OUTPUT_UNIT \
--preprocess_mode $PREPROCESS_MODE \
--vocab_size $VOCAB_SIZE \
