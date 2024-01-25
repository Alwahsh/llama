# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List
from time_measure import TimeMeasure
import pdb

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 123124124124,
    max_batch_size: int = 4,
    disable_eos: bool = False,
    in_seq_len: int = 1,
    compression_type: int = -1, # -1 = normal code without considering compression at all. 0 = perform needed conversions but don't compress. 1 = perform needed operations and compress.
    compression_attribute: int = 10,
    warmup_iterations: int = 10,
    measured_iterations: int = 10,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        disable_eos=disable_eos,
        compression_type = compression_type,
        compression_attribute = compression_attribute,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        # "I believe the meaning of life is",
        # "Simply put, the theory of relativity states that ",
        # """A brief message congratulating the team on the launch:

        # Hi everyone,
        
        # I just """,
        # # Few shot prompt (providing a few examples before asking model to complete more);
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
        "it is my destiny",
        # "The answer to life, the universe, and everything is the answer to life universe",
        # "The story begins with a poor boy living in a country with a stunning princess",
        # "it is my destiny",
    ]
    text = ""
    for _ in range(1,in_seq_len):
        text += "it"
    text = text.strip()
    prompts = [text] * max_batch_size
    # Repeat the completion 10 times first.
    for _ in range(warmup_iterations):
        generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
    tm = TimeMeasure()
    for i in range(measured_iterations):
        tm.set_prefix('llama7b')
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            tm=tm,
        )
        # print(f"Results are {results}")
        # pdb.set_trace()
        times = tm.all_times()
        with open(f'time_prefill_{i}.txt', 'w') as file:
            file.write(str(times['llama7b']['prefill'][0]))
        with open(f'time_decode_{i}.txt', 'w') as file:
            file.write(str(sum(times['llama7b']['decode'])))
        tm.reset_stats()
        with open(f'response_{i}.txt', 'w') as file:
            file.write(results[0]["generation"].replace("\n","\\n"))
    # pdb.set_trace()
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
