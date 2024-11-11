#!/bin/bash
#SBATCH -o solar_results.out
#SBATCH -J solar

model_name=delayformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path Solar/solar_AL.txt \
  --model_id solar_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 7 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe learnable_pe \
  --des 'Exp' \
  --itr 1 \
  --n_vars 137 \
  --learning_rate 0.0005 \
  --n 49 \
  --p1 24 \
  --p2 7

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path Solar/solar_AL.txt \
  --model_id solar_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 7 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe learnable_pe \
  --des 'Exp' \
  --itr 1 \
  --n_vars 137 \
  --learning_rate 0.0005 \
  --n 49 \
  --p1 24 \
  --p2 7
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path Solar/solar_AL.txt \
  --model_id solar_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 7 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe learnable_pe \
  --des 'Exp' \
  --itr 1 \
  --n_vars 137 \
  --learning_rate 0.0005 \
  --n 49 \
  --p1 24 \
  --p2 7
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ \
  --data_path Solar/solar_AL.txt \
  --model_id solar_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --lradj type1 \
  --e_layers 2 \
  --n_heads 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 7 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 512 \
  --pe learnable_pe \
  --des 'Exp' \
  --itr 1 \
  --n_vars 137 \
  --learning_rate 0.0005 \
  --n 49 \
  --p1 24 \
  --p2 7