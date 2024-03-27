python main.py \
--eval_data_name cais/mmlu \
--eval_data_config_name all \
--eval_set_key test \
--text_key question \
--label_key answer_text \
--n_eval_data_points 1000 \
--num_proc 1 \
--method guided-prompting \
--openai_creds_key_file "openai_creds/openai_api_key.txt" \
--local_api_type "openai" \
--guided_prompting_task_type QA \
--use_local_model