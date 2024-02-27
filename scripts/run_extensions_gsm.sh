DATASET='gsm'
MODEL='phi-2'
RANGE_START=$1
RANGE_END=$2
TAG='deepseek'


# plan
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode plan_v2

#meta_eval_cot
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode metacognitive_eval_cot --data_path outputs/gsm/cot/minibatches/phi_cot_${RANGE_START}_${RANGE_END}.jsonl

# tot
python src/solve_tot.py --tag phi --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode tot

#meta_eval_tot
python src/solve_tot.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode metacognitive_eval_tot --data_path outputs/gsm/tot/minibatches/phi_tot_${RANGE_START}_${RANGE_END}.jsonl
