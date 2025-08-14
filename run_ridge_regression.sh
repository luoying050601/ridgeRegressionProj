# shellcheck disable=SC2006
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"
# shellcheck disable=SC2034
#model_type='brainbert'
nohup python3 ridge.py \
--type 'brainbert' \
--arpha 0.1 \
>>"create_word_embedding_${cur_date}".out 2>&1 &

#nohup python3 ridge.py \
#--type 'bert' \
#--arpha 0.1 \
#>>"ridge_regression_bert_${cur_date}".out 2>&1 &
