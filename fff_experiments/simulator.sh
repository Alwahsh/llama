#!/bin/bash

# Source and destination filenames
source_file="params_templates/params13b.template"

output_csv="simulated13b_results.csv"

mp_val=2

simulated_dir="simulated13b"
mkdir -p $simulated_dir

destination_file="$simulated_dir/params.json"

num_iterations=10

fff_depths=(-1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

batch_sizes=(1 2 4 8 16 32 64 128)

echo "batch_size,fff_depth,time" > $output_csv


for ((i=1; i<=$num_iterations; i++)); do
    for batch_size in "${batch_sizes[@]}"; do
        # Simulate for each FFF Depth
        for fff_depth in "${fff_depths[@]}"; do
            # Copy the file
            cp "$source_file" "$destination_file"

            # Replace FFF Depth with the actual value
            sed -i "s/TEMP_FFF_DEPTH/$fff_depth/g" "$destination_file"

            torchrun --nproc_per_node $mp_val ./../example_text_completion.py --ckpt_dir $simulated_dir/ --tokenizer_path ./../tokenizer.model --max_seq_len 1024 --max_batch_size $batch_size
            measured_time=$(cat time.txt)
            echo "-1" > time.txt
            echo "$batch_size,$fff_depth,$measured_time" >> $output_csv
        done
    done
done
