DATASET='gsm'
MODEL='phi-2'
RANGE_START=$1
RANGE_END=$2
TAG=$3

# cot
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode cot
# eot
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode eot
# check_eot
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode check_eot --data_path outputs/gsm/eot/${TAG}_eot_${RANGE_START}_${RANGE_END}.jsonl
# pot
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode pot
# check_pot
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode check_pot --data_path outputs/gsm/pot/${TAG}_pot_${RANGE_START}_${RANGE_END}.jsonl
# plan
python src/solve.py --tag ${TAG} --range_start ${RANGE_START} --range_end ${RANGE_END} --dataset ${DATASET} --model ${MODEL} --mode plan 
