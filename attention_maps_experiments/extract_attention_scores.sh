input_file="sample_inputs.txt"
# Read the input file line by line
while IFS= read -r line; do
    # Create a new file called "sample_input.txt" with the current line
    echo "$line" > sample_input.txt

    torchrun --nproc_per_node 1 ../example_text_completion.py --ckpt_dir ../llama-2-7b --tokenizer_path ../tokenizer.model --max_seq_len 16 --in_seq_len 16 --max_gen_len 1 --max_batch_size 1 --disable_eos=1 --warmup_iterations=1 --measured_iterations=0

done < "$input_file"