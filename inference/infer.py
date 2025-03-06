import vllm
import argparse
import os 
import json 
import utils.utils as utils
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer
import transformers
import torch
from pprint import pprint
import time
from fvcore.nn.flop_count import FlopCountAnalysis


def load_data(args, filename):


    lines = open(os.path.join(args.data_path, filename), encoding='utf-8').readlines()
    lines = [json.loads(x) for x in lines if x.strip()]
    list_data_dict =  lines 
    
    if 'qwen' in args.base_model.lower() or 'qw2' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # list_data_dict = list_data_dict[:10]
        for example in list_data_dict:
        
            prompt = '<|im_start|>'+'user\n'+ example["instruction"] +'<|im_end|>\n<|im_start|>assistant\n'

            # prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' + \
            # '<|im_start|>'+'user\n'+ example["instruction"] +'<|im_end|>\n<|im_start|>assistant\n'
        
            prompts.append(prompt)
        print('qwen2:', prompts[0])

    elif 'llama-3' in args.base_model.lower() or 'dpsk' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        for example in list_data_dict:

            prompt = example['instruction']
            prompts.append(prompt)
        print('llama3:', prompts[0])

    elif 'flan-t5' in args.base_model.lower():
        prompts = []
        tokenizer = T5Tokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            # Format appropriate for T5
            # T5 was fine-tuned to respond to prefixed tasks
            table_str = json.dumps(example.get('table', {}))
            instruction = example.get('instruction', example.get('question', ''))
            
            # Format differently based on task type
            if 'visualization' in example.get('qtype', '').lower():
                prompt = f"create chart: {instruction}\ntable: {table_str}"
            else:
                prompt = f"answer question: {instruction}\ntable: {table_str}"
                
            prompts.append(prompt)
        print('flan-t5 prompt example:', prompts[0])

    assert len(prompts) == len(list_data_dict)
    return prompts, list_data_dict, tokenizer

def run(args):

    sampling_params = vllm.SamplingParams(n = args.sample_n, temperature=args.temperature, top_p=0.95, max_tokens=8000)

    print("args:", args)
    model = vllm.LLM(model=args.base_model, tensor_parallel_size=2, trust_remote_code=True)

    fnames = [x for x in os.listdir(args.data_path) if x.endswith('.jsonl')]
    for filename in fnames:
        print(filename)
        prompts, raw_datas, tokenizer = load_data(args, filename)
        print(args.temperature)

        # Add FLOPS tracking
        flops_results = {}
        inference_times = {}
        
        # Track FLOPS for a sample batch
        if args.measure_flops:
            if 'flan-t5' in args.base_model.lower():
                # For T5 models
                sample_input = tokenizer(prompts[0], return_tensors="pt").to(args.device)
                flops = FlopCountAnalysis(model, sample_input)
                batch_flops = flops.total()
                # Estimate total FLOPS
                total_flops = batch_flops * len(prompts)
                flops_results[filename] = total_flops
                
                # Measure inference time
                start_time = time.time()
            
        outputs = model.generate(prompts, sampling_params)

        assert len(outputs) == len(raw_datas)
        
        for idx, output in enumerate(outputs):
            prompt = output.prompt
            generated_texts = [item.text for item in output.outputs ]

            raw_datas[idx]["prediction"] = generated_texts


        save_path = os.path.join(args.outdir, args.base_model.split('/')[-1]+'_'+filename.split('.')[0]+'.jsonl')

        with open(save_path, 'w') as f:
            for item in raw_datas:
                f.write(json.dumps(item)+'\n')

        if args.measure_flops:
            end_time = time.time()
            inference_times[filename] = end_time - start_time
            
    # Save FLOPS results if measured
    if args.measure_flops:
        flops_report = {
            "model": args.base_model,
            "per_file_flops": flops_results,
            "total_flops": sum(flops_results.values()),
            "inference_times": inference_times,
            "flops_per_second": sum(flops_results.values()) / sum(inference_times.values())
        }
        with open(f"{args.outdir}/flops_report.json", "w") as f:
            json.dump(flops_report, f, indent=2)


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--temperature", default=0.0, type=float, help="config path")
    parser.add_argument("--task", default="complete", type=str, help="config path")
    parser.add_argument("--outdir", default="outputs_size", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    parser.add_argument("--model_max_length", type=int, default=8000, help="beam size")
    parser.add_argument("--sample_n", type=int, default=1, help="beam size")
    parser.add_argument("--measure_flops", type=bool, default=False, help="measure FLOPS")

    args = parser.parse_args()

    run(args)
