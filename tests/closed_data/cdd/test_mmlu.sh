# Get the options
while getopts ":p:m:" option; do
   case $option in
      p) # port number
         port=$OPTARG;;
      m) # Enter closed_data name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name
echo "local port: ", $port

# test guided prompting closed_data contamination method
python main.py \
--eval_data_name cais/mmlu \
--eval_data_config_name all \
--eval_set_key test \
--text_key question \
--label_key answer_text \
--n_eval_data_points 100 \
--num_proc 16 \
--method cdd \
--model_name $model_name \
--local_port $port \
--num_samples 20 \
--temperature 0.8 \
--cdd_alpha 0.05 \
--cdd_xi 0.01