from human_eval.data import write_jsonl, read_problems
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch
from core import fix_indents, filter_code
import os
from tqdm import tqdm
import typing
from human_eval.evaluation import evaluate_functional_correctness

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]
def run_eval(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_samples_per_task: int,
        out_path: str,
        generate_batch_completion: BatchGenerator,
        format_tabs: bool = False,
        task: str = 'humaneval',
):
    if task == 'humaneval':
        problems = read_problems()
    elif task == 'humanevalplus':
        problems = read_problems('results/humanevalplus/test.jsonl')
    elif task == 'mbpp':
        problems = read_problems('results/mbpp/test.jsonl')
        # print(problems)
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    for task_id in problems:

        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        prompt = code_anchor(prompt)
        print(prompt)
        if task_id != 482:
            batch_completions = generate_batch_completion(
                model, tokenizer, prompt, num_samples_per_task
            )
        else:
            batch_completions = ['']
        print(batch_completions[0])
        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]
        pbar.update(num_samples_per_task)

    write_jsonl(out_path, samples)


@torch.inference_mode()
def generate_batch_completion(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)
    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(generated_ids[0][input_ids_cutoff:])
    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True
    )
    return [filter_code(fix_indents(completion)) for completion in batch_completions]

def code_anchor(example):
    example = example.replace('\n', '\n<ANCHOR>')
    return example


if __name__ == "__main__":
    # adjust for n = 10 etc
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--model')

    args = parser.parse_args()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    num_samples_per_task = 1
    task = args.task
    model = args.model
    out_path = "results/" + task + ".jsonl"
    os.makedirs("results/"+task, exist_ok=True)
    tokenizer = LlamaTokenizer.from_pretrained(
        model, use_fast=False, legacy=False,
    )
    model = LlamaForCausalLM.from_pretrained(
        model,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
    import time
    start_time = time.time()
    with torch.no_grad():
        run_eval(
            model,
            tokenizer,
            num_samples_per_task,
            out_path,
            generate_batch_completion,
            True,
            task,
        )
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Time elapsed: ", time_elapsed)
    if task == 'humanevalplus':
        res = evaluate_functional_correctness(sample_file=out_path,
                                          problem_file='results/humanevalplus/test.jsonl')
        print(res)
    elif task == 'mbpp':
        res = evaluate_functional_correctness(sample_file=out_path, problem_file='results/mbpp/test.jsonl')
        print(res)

    elif task == 'humaneval':
        res = evaluate_functional_correctness(sample_file=out_path)
        print(res)

