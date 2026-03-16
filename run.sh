#!bin/bash


for i in {1..3}
do
    echo "Running chunk $i"
    python3 string_model_gen_trajs_Tmax.py $i $(( (i-1) * 50 )) 50
done;


