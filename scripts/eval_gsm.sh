RANGE_START=$1
RANGE_END=$2
#TAG=$3

python src/analyze.py \
  --plan outputs/gsm/plan/deepseek_plan_rating_${RANGE_START}_${RANGE_END}.jsonl \
  --cot outputs/gsm/cot/phi_cot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot outputs/gsm/pot/phi_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot outputs/gsm/eot/phi_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot_assertion outputs/gsm/check_pot/phi_check_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot_assertion outputs/gsm/check_eot/phi_check_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --metacognitive_eval outputs/gsm/metacognitive_eval_cot/deepseek_metacognitive_eval_cot_${RANGE_START}_${RANGE_END}.jsonl \
  --tag gsm_analysis
