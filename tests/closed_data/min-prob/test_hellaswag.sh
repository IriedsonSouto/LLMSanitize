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

# test min-K-prob closed_data contamination method
python main.py \
--eval_data_name Rowan/hellaswag \
--eval_set_key test \
--text_keys ctx+endings+label \
--n_eval_data_points 100 \
--num_proc 16 \
--method min-prob \
--local_port $port \
--model_name $model_name \
--max_output_tokens 1 \
--top_logprobs 5 \
--max_request_time 10 \
--echo \
--minkprob_local_port_2 $port \
--minkprob_model_name_2 $model_name
