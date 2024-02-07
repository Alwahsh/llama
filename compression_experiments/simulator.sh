#!/bin/bash

# Get current timestamp
timestamp=$(date +%s)
mkdir -p "${timestamp}_results"
output_dir="${timestamp}_results"
output_csv="${output_dir}/results.csv"

mp_val=2

simulated_dir="./../llama-2-13b"

destination_file="$simulated_dir/params.json"

num_iterations=1

compression_types=(2 3 4 1)
# compression_types=(1)

batch_sizes=(1)

gen_lens=(-1)

in_seq_lens=(14 30 126 254 510 1022 2046 4094)

# No compression
compression_attributes_0=(0)
# Lossless
compression_attributes_1=(0)
# Precision
compression_attributes_2=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# Rate
compression_attributes_3=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# Tolerance
compression_attributes_4=(1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 5 10 50 100 500 1000 5000 10000)

fff_depth=-1

warmup_iterations=0

measured_iterations=1

echo "id,batch_size,compression_type,compression_attribute,gen_len,in_seq_len,time_prefill,time_decode,k_size,v_size,response" > $output_csv
id=0
for ((i=1; i<=$num_iterations; i++)); do
    for compression_type in "${compression_types[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            # Loop over gen_len values
            for gen_len_temp in "${gen_lens[@]}"; do
                # Loop over in_seq_len values
                for in_seq_len in "${in_seq_lens[@]}"; do
                    if ((gen_len_temp == -1)); then
                        gen_len=$((in_seq_len + 2))
                    else
                        gen_len=$gen_len_temp
                    fi
                    # Only simulate if in_seq_len is smaller than gen_len
                    if ((gen_len - in_seq_len >= 2)); then
                        # Simulate for each FFF Depth
                            # Copy the file
                            # cp "$source_file" "$destination_file"
                        compression_attributes_array_name="compression_attributes_$compression_type"
                        eval "compression_attributes=\${$compression_attributes_array_name[*]}"

                        for compression_attribute in $compression_attributes; do
                            # Replace FFF Depth with the actual value
                            sed -i "s/TEMP_FFF_DEPTH/$fff_depth/g" "$destination_file"
                            id=$((id+1))
                            torchrun --nproc_per_node $mp_val ./../example_text_completion.py --ckpt_dir $simulated_dir/ --tokenizer_path ./../tokenizer.model --max_seq_len $gen_len --max_gen_len $gen_len --max_batch_size $batch_size --disable_eos 0 --in_seq_len $in_seq_len --compression_type $compression_type --compression_attribute $compression_attribute --warmup_iterations $warmup_iterations --measured_iterations $measured_iterations
                            for ((k=0; k<$measured_iterations; k++)); do
                                measured_time_prefill=$(cat time_prefill_$k.txt)
                                measured_time_decode=$(cat time_decode_$k.txt)
                                echo "-1" > time_prefill_$k.txt
                                echo "-1" > time_decode_$k.txt
                                k_size=$(du -b k_cache.pkl | awk '{print $1}')
                                v_size=$(du -b v_cache.pkl | awk '{print $1}')
                                response=$(cat response_$k.txt)
                                echo "-1" > response_$k.txt
                                echo "$id,$batch_size,$compression_type,$compression_attribute,$gen_len,$in_seq_len,$measured_time_prefill,$measured_time_decode,$k_size,$v_size,$response" >> $output_csv
                                cp ./caching_statistics_$k.json $output_dir/caching_statistics_${id}_${k}.json
                                echo "{}" > ./caching_statistics_$k.json
                            done
                        done
                    fi
                done
            done
        done
    done
done
