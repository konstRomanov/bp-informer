python -m bp-informer.main_informer \
--model informer \
--data Stock \
--root_path "./bp-informer/data/stock" \
--data_path "Stock.csv" \
--features MS \
--ftr_num 2 \
--d_out 1 \
--target price \
--freq "15t" \
--seq_len 48 \
--pred_len 12 \
--batch_size 12 \
--embed t2v \
--inverse \
--predict
