python -u run.py \
  --is_training 1 \
  --model_id Industry_10_5 \
  --model Informer \
  --data custom \
  --features M \
  --seq_len 10 \
  --label_len 5 \
  --pred_len 5 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --gpu 0 \
  --enc_in 118 \
  --dec_in 118 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1
  --features 'MS'
  &

python -u get_main.py \
  --model_id Industry_10_5 \
  --model Transformer \
  --data custom \
  --features M \
  --seq_len 10 \
  --label_len 5 \
  --pred_len 5 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --gpu 0 \
  --enc_in 118 \
  --dec_in 118 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 &

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model Transformer \
  --data custom \
  --features M \
  --seq_len 10 \
  --label_len 5 \
  --pred_len 5 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --gpu 0 \
  --enc_in 118 \
  --dec_in 118 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 &

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model Transformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --gpu 7 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 &
