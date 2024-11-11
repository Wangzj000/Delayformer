#!/bin/bash
#SBATCH -o ecl_results.out
#SBATCH -J ecl

model_name=delayformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path electricity/electricity.csv \
  --model_id ecl_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 7 \
  --batch_size 32 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe fix_pe \
  --temporal_encoding \
  --des 'Exp' \
  --itr 1 \
  --n_vars 321 \
  --learning_rate 0.0005 \
  --n 49 \
  --p1 24 \
  --p2 7

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path electricity/electricity.csv \
  --model_id ecl_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 7 \
  --batch_size 32 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe fix_pe \
  --temporal_encoding \
  --des 'Exp' \
  --itr 1 \
  --n_vars 321 \
  --learning_rate 0.0005 \
  --n 49 \
  --p1 24 \
  --p2 7

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path electricity/electricity.csv \
  --model_id ecl_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 7 \
  --batch_size 32 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe fix_pe \
  --temporal_encoding \
  --des 'Exp' \
  --itr 1 \
  --n_vars 321 \
  --learning_rate 0.0005 \
  --n 49 \
  --p1 24 \
  --p2 7

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path electricity/electricity.csv \
  --model_id ecl_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 7 \
  --batch_size 32 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe fix_pe \
  --temporal_encoding \
  --des 'Exp' \
  --itr 1 \
  --n_vars 321 \
  --learning_rate 0.0001 \
  --n 57 \
  --p1 10 \
  --p2 19