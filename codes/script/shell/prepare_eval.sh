echo 'write output of baselines under models/'
python codes/script/write_score_files.py

echo 'writing affinity matrix'
echo wea val

echo "all single multiple" | xargs -n 1 -r -P 3 python codes/script/preprocess/precompute_affinity_matrix.py  --pair_score_file models/wea/res_val.csv --method wea --split val --eval_type

echo ${DIR} test
echo "all single multiple" | xargs -n 1 -r -P 3 python codes/script/preprocess/precompute_affinity_matrix.py  --pair_score_file models/wea/res_test.csv --method wea --split test --eval_type

for DIR in `ls models`
do

echo ${DIR} val

echo "all single multiple" | xargs -n 1 -r -P 3 python codes/script/preprocess/precompute_affinity_matrix.py  --pair_score_file models/${DIR}/res_val.csv --method ${DIR} --split val --eval_type

echo ${DIR} test
echo "all single multiple" | xargs -n 1 -r -P 3 python codes/script/preprocess/precompute_affinity_matrix.py  --pair_score_file models/${DIR}/res_test.csv --method ${DIR} --split test --eval_type
done