RANGE_START=$1
RANGE_END=$2
#TAG=$3

# python src/analyze_test.py \
#   --plan outputs/gsm/plan/${TAG}_plan_${RANGE_START}_${RANGE_END}.jsonl \
#   --cot outputs/gsm/cot/${TAG}_cot_${RANGE_START}_${RANGE_END}.jsonl \
#   --pot outputs/gsm/pot/${TAG}_pot_${RANGE_START}_${RANGE_END}.jsonl \
#   --eot outputs/gsm/eot/${TAG}_eot_${RANGE_START}_${RANGE_END}.jsonl \
#   --pot_assertion outputs/gsm/check_pot/${TAG}_check_pot_${RANGE_START}_${RANGE_END}.jsonl \
#   --eot_assertion outputs/gsm/check_eot/${TAG}_check_eot_${RANGE_START}_${RANGE_END}.jsonl \
#   --tag gsm_analysis

# python src/analyze_extension1.py \
#   --plan outputs/gsm/plan/deepseek_plan_rating_${RANGE_START}_${RANGE_END}.jsonl \
#   --cot outputs/gsm/cot/phi_cot_${RANGE_START}_${RANGE_END}.jsonl \
#   --pot outputs/gsm/pot/phi_pot_${RANGE_START}_${RANGE_END}.jsonl \
#   --eot outputs/gsm/eot/phi_eot_${RANGE_START}_${RANGE_END}.jsonl \
#   --pot_assertion outputs/gsm/check_pot/phi_check_pot_${RANGE_START}_${RANGE_END}.jsonl \
#   --eot_assertion outputs/gsm/check_eot/phi_check_eot_${RANGE_START}_${RANGE_END}.jsonl \
#   --metacognitive_eval outputs/gsm/metacognitive_eval_cot/deepseek_metacognitive_eval_cot_${RANGE_START}_${RANGE_END}.jsonl \
#   --tag gsm_analysis

  
python src/analyze_extension2.py \
  --plan outputs/gsm/plan/deepseek_plan_rating_${RANGE_START}_${RANGE_END}.jsonl \
  --tot outputs/gsm/tot/phi_tot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot outputs/gsm/pot/phi_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot outputs/gsm/eot/phi_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot_assertion outputs/gsm/check_pot/phi_check_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot_assertion outputs/gsm/check_eot/phi_check_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --metacognitive_eval outputs/gsm/metacognitive_eval_tot/deepseek_metacognitive_eval_tot_${RANGE_START}_${RANGE_END}.jsonl \
  --tag gsm_analysis