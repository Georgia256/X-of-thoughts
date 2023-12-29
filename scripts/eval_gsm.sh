RANGE_START = 0
RANGE_END = 100

python src/analyze.py \
  --plan outputs/gsm/plan/demo_plan_${RANGE_START}_${RANGE_END}.jsonl \
  --cot outputs/gsm/cot/demo_cot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot outputs/gsm/pot/demo_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot outputs/gsm/eot/demo_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot_assertion outputs/gsm/check_pot/demo_check_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot_assertion outputs/gsm/check_eot/demo_check_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --tag gsm_analysis