#!/bin/bash

# Get current timestamp
timestamp=$(date +%s)
mkdir -p "${timestamp}_results"
output_csv="${timestamp}_results/results.csv"

mp_val=1

simulated_dir="./../llama-2-7b"

num_iterations=1

use_cpu=0

fff_depths=(-1 0)

batch_sizes=(1 16 32 64)

# gen_lens=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
gen_lens=(2048)

# in_seq_lens=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
in_seq_lens=(1 2 4 8 16 32 64 128 256 512 1024 2046)

warmup_iterations=10
measured_iterations=10

echo "batch_size,fff_depth,gen_len,in_seq_len,time_prefill,time_decode,prefill_time_attention_rms_norm,prefill_time_ffn_rms_norm,prefill_time_transformer_rms_norm,prefill_time_attention,prefill_time_ffn,prefill_time_transformer_block,prefill_time_transformer,decode_time_attention_rms_norm,decode_time_ffn_rms_norm,decode_time_transformer_rms_norm,decode_time_attention,decode_time_ffn,decode_time_transformer_block,decode_time_transformer" > $output_csv


for ((i=1; i<=$num_iterations; i++)); do
    # Simulate for each FFF Depth
    for fff_depth in "${fff_depths[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            # Loop over gen_len values
            for gen_len in "${gen_lens[@]}"; do
                # Loop over in_seq_len values
                for in_seq_len in "${in_seq_lens[@]}"; do
                    # Only simulate if in_seq_len is smaller than gen_len
                    if ((gen_len - in_seq_len >= 2)); then
                        torchrun --nproc_per_node $mp_val ./../example_text_completion.py --ckpt_dir $simulated_dir/ --tokenizer_path ./../tokenizer.model --max_seq_len $gen_len --max_gen_len $gen_len --max_batch_size $batch_size --disable_eos 1 --in_seq_len $in_seq_len --fff_depth $fff_depth --warmup_iterations $warmup_iterations --measured_iterations $measured_iterations --use_cpu $use_cpu
                        for ((k=0; k<$measured_iterations; k++)); do
                            measured_time_prefill=$(cat time_prefill_$k.txt)
                            measured_time_decode=$(cat time_decode_$k.txt)
                            measured_prefill_time_attention_rms_norm=$(cat prefill_time_attention_rms_norm_$k.txt)
                            measured_prefill_time_ffn_rms_norm=$(cat prefill_time_ffn_rms_norm_$k.txt)
                            measured_prefill_time_transformer_rms_norm=$(cat prefill_time_transformer_rms_norm_$k.txt)
                            measured_prefill_time_attention=$(cat prefill_time_attention_$k.txt)
                            measured_prefill_time_ffn=$(cat prefill_time_ffn_$k.txt)
                            measured_prefill_time_transformer_block=$(cat prefill_time_transformer_block_$k.txt)
                            measured_prefill_time_transformer=$(cat prefill_time_transformer_$k.txt)
                            measured_decode_time_attention_rms_norm=$(cat decode_time_attention_rms_norm_$k.txt)
                            measured_decode_time_ffn_rms_norm=$(cat decode_time_ffn_rms_norm_$k.txt)
                            measured_decode_time_transformer_rms_norm=$(cat decode_time_transformer_rms_norm_$k.txt)
                            measured_decode_time_attention=$(cat decode_time_attention_$k.txt)
                            measured_decode_time_ffn=$(cat decode_time_ffn_$k.txt)
                            measured_decode_time_transformer_block=$(cat decode_time_transformer_block_$k.txt)
                            measured_decode_time_transformer=$(cat decode_time_transformer_$k.txt)
                            echo "-1" > time_prefill_$k.txt
                            echo "-1" > time_decode_$k.txt
                            echo "-1" > prefill_time_attention_rms_norm_$k.txt
                            echo "-1" > prefill_time_ffn_rms_norm_$k.txt
                            echo "-1" > prefill_time_transformer_rms_norm_$k.txt
                            echo "-1" > prefill_time_attention_$k.txt
                            echo "-1" > prefill_time_ffn_$k.txt
                            echo "-1" > prefill_time_transformer_block_$k.txt
                            echo "-1" > prefill_time_transformer_$k.txt
                            echo "-1" > decode_time_attention_rms_norm_$k.txt
                            echo "-1" > decode_time_ffn_rms_norm_$k.txt
                            echo "-1" > decode_time_transformer_rms_norm_$k.txt
                            echo "-1" > decode_time_attention_$k.txt
                            echo "-1" > decode_time_ffn_$k.txt
                            echo "-1" > decode_time_transformer_block_$k.txt
                            echo "-1" > decode_time_transformer_$k.txt
                            echo "$batch_size,$fff_depth,$gen_len,$in_seq_len,$measured_time_prefill,$measured_time_decode,$measured_prefill_time_attention_rms_norm,$measured_prefill_time_ffn_rms_norm,$measured_prefill_time_transformer_rms_norm,$measured_prefill_time_attention,$measured_prefill_time_ffn,$measured_prefill_time_transformer_block,$measured_prefill_time_transformer,$measured_decode_time_attention_rms_norm,$measured_decode_time_ffn_rms_norm,$measured_decode_time_transformer_rms_norm,$measured_decode_time_attention,$measured_decode_time_ffn,$measured_decode_time_transformer_block,$measured_decode_time_transformer" >> $output_csv
                        done
                    fi
                done
            done
        done
    done
done
