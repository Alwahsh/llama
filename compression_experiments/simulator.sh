#!/bin/bash

# Source and destination filenames
source_file="params_templates/params7b.template"

output_csv="simulated7b_results.csv"

mp_val=1

simulated_dir="simulated7b"
mkdir -p $simulated_dir

destination_file="$simulated_dir/params.json"

num_iterations=1

compression_types=(0 1)

batch_sizes=(1 2 4 8 16 32 64)

gen_lens=(1 2 4 8 2048)

in_seq_lens=(1 2 4 8 2048)

fff_depth=-1

echo "batch_size,compression_type,gen_len,in_seq_len,time_prefill,time_decode" > $output_csv


for ((i=1; i<=$num_iterations; i++)); do
    for batch_size in "${batch_sizes[@]}"; do
        # Loop over gen_len values
        for gen_len in "${gen_lens[@]}"; do
            # Loop over in_seq_len values
            for in_seq_len in "${in_seq_lens[@]}"; do
                # Only simulate if in_seq_len is smaller than gen_len
                if ((gen_len - in_seq_len >= 2)); then
                    # Simulate for each FFF Depth
                    for compression_type in "${compression_types[@]}"; do
                        # Copy the file
                        cp "$source_file" "$destination_file"

                        # Replace FFF Depth with the actual value
                        sed -i "s/TEMP_FFF_DEPTH/$fff_depth/g" "$destination_file"

                        torchrun --nproc_per_node $mp_val ./../example_text_completion.py --ckpt_dir $simulated_dir/ --tokenizer_path ./../tokenizer.model --max_seq_len $gen_len --max_gen_len $gen_len --max_batch_size $batch_size --disable_eos 1 --in_seq_len $in_seq_len --compression_type $compression_type
                        for ((k=0; k<10; k++)); do
                            measured_time_prefill=$(cat time_prefill_$k.txt)
                            measured_time_decode=$(cat time_decode_$k.txt)
                            echo "-1" > time_prefill_$k.txt
                            echo "-1" > time_decode_$k.txt
                            echo "$batch_size,$compression_type,$gen_len,$in_seq_len,$measured_time_prefill,$measured_time_decode" >> $output_csv
                        done
                    done
                fi
            done
        done
    done
done


