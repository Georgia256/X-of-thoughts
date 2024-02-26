RANGE_START=$1
RANGE_END=$2
TAG='phi'


python src/analyze_extension2a.py \
  --plan outputs/gsm/plan/${TAG}_plan_${RANGE_START}_${RANGE_END}.jsonl \
  --tot outputs/gsm/tot/${TAG}_tot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot outputs/gsm/pot/${TAG}_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot outputs/gsm/eot/${TAG}_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot_assertion outputs/gsm/check_pot/${TAG}_check_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot_assertion outputs/gsm/check_eot/${TAG}_check_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --tag gsm_analysis