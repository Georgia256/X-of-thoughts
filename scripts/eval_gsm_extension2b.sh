RANGE_START=0
RANGE_END='end'


python src/analyze_extension2b.py \
  --plan outputs/gsm/plan/deepseek_plan_rating_${RANGE_START}_${RANGE_END}.jsonl \
  --tot outputs/gsm/tot/phi_tot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot outputs/gsm/pot/phi_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot outputs/gsm/eot/phi_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --pot_assertion outputs/gsm/check_pot/phi_check_pot_${RANGE_START}_${RANGE_END}.jsonl \
  --eot_assertion outputs/gsm/check_eot/phi_check_eot_${RANGE_START}_${RANGE_END}.jsonl \
  --metacognitive_eval outputs/gsm/metacognitive_eval_tot/deepseek_metacognitive_eval_tot_${RANGE_START}_${RANGE_END}.jsonl \
  --tag gsm_analysis