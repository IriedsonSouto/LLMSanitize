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
--eval_data_name Rowan/hellaswag \
--eval_set_key validation \
--text_key ctx \
--label_key activity_label \
--n_eval_data_points 100 \
--num_proc 16 \
--method guided-prompting \
--local_port $port \
--model_name $model_name \
--guided_prompting_task_type NLI \
--use_local_model