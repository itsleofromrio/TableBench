import argparse
import os
import json
import utils.utils as utils
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
import torch
from pprint import pprint


def load_data(args, filename):
    lines = open(os.path.join(args.data_path, filename), encoding='utf-8').readlines()
    lines = [json.loads(x) for x in lines if x.strip()]
    list_data_dict = lines
    
    if 'qwen' in args.base_model.lower() or 'qw2' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            prompt = '<|im_start|>'+'user\n'+ example["instruction"] +'<|im_end|>\n<|im_start|>assistant\n'
            prompts.append(prompt)
        print('qwen2:', prompts[0])

    elif 'llama-3' in args.base_model.lower() or 'dpsk' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            prompt = example['instruction']
            prompts.append(prompt)
        print('llama3:', prompts[0])
        
    elif 'deepseek-coder-v2' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        for example in list_data_dict:
            # Format messages for DeepSeek-Coder-V2 chat template
            table_str = json.dumps(example.get('table', {}))
            instruction = example.get('instruction', example.get('question', ''))
            
            if 'instruct' in args.base_model.lower():
                # For instruct models, use the chat template
                messages = [
                    {'role': 'user', 'content': f"{instruction}\nHere is the table data: {table_str}"}
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=None)
            else:
                # For base models, use a simpler format
                prompt = f"\n{instruction}\nHere is the table data: {table_str}\n"
            
            prompts.append(prompt)
        print('deepseek-coder-v2:', prompts[0])
        
    elif 'phi-2' in args.base_model.lower():
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
            # Phi-2 works well with a simple instruction format
            table_str = json.dumps(example.get('table', {}))
            instruction = example.get('instruction', example.get('question', ''))
            prompt = f"Instruction: {instruction}\nTable: {table_str}\n\nResponse:"
            prompts.append(prompt)
        print('phi-2:', prompts[0])
        
    elif 'flan-t5' in args.base_model.lower():
        prompts = []
        tokenizer = T5Tokenizer.from_pretrained(args.base_model)
        for example in list_data_dict:
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
    print("args:", args)
    
    # Get list of test files
    fnames = [x for x in os.listdir(args.data_path) if x.endswith('.jsonl')]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.outdir), exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for filename in fnames:
        print(f"Processing {filename}")
        prompts, raw_datas, tokenizer = load_data(args, filename)
        print(f"Temperature: {args.temperature}")
        
        # Load the appropriate model based on model type
        if 'flan-t5' in args.base_model.lower():
            model = T5ForConditionalGeneration.from_pretrained(args.base_model).to(device)
            
            # Process in batches
            batch_size = 8  # Adjust based on your memory constraints
            all_outputs = []
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=args.model_max_length,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 0.01) if args.temperature > 0 else 1.0,
                        num_return_sequences=args.sample_n
                    )
                
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_outputs.extend(decoded)
            
            # Format outputs to match expected structure
            for idx, output in enumerate(all_outputs):
                if idx < len(raw_datas):
                    raw_datas[idx]["prediction"] = [output]  # Match vLLM format with list
            
        elif 'deepseek-coder-v2' in args.base_model.lower():
            # DeepSeek-Coder-V2 requires trust_remote_code=True
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)
            
            # Process in batches - smaller batch size for MoE models
            batch_size = 2  # Smaller batch size for MoE models
            all_outputs = []
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.model_max_length,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 0.01) if args.temperature > 0 else 1.0,
                        num_return_sequences=args.sample_n,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # For DeepSeek-Coder-V2, extract the assistant's response
                for j, prompt in enumerate(batch_prompts):
                    if i*batch_size + j < len(decoded):
                        if 'instruct' in args.base_model.lower():
                            # For instruct models, find the assistant's response
                            assistant_idx = decoded[j].find("Assistant:")
                            if assistant_idx != -1:
                                decoded[j] = decoded[j][assistant_idx + len("Assistant:"):].strip()
                        else:
                            # For base models, remove the prompt
                            decoded[j] = decoded[j][len(prompt):].strip()
                
                all_outputs.extend(decoded)
            
            # Format outputs to match expected structure
            for idx, output in enumerate(all_outputs):
                if idx < len(raw_datas):
                    raw_datas[idx]["prediction"] = [output]
        
        else:
            # For other models (Llama, Qwen, Phi, etc.)
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)
            
            # Process in batches
            batch_size = 4 if 'phi-2' in args.base_model.lower() else 8
            all_outputs = []
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.model_max_length,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 0.01) if args.temperature > 0 else 1.0,
                        num_return_sequences=args.sample_n
                    )
                
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # For non-encoder-decoder models, we need to remove the prompt from the output
                for j, prompt in enumerate(batch_prompts):
                    if i*batch_size + j < len(all_outputs):
                        prompt_tokens = len(tokenizer.encode(prompt))
                        decoded[j] = decoded[j][len(tokenizer.decode(tokenizer.encode(prompt)[:prompt_tokens])):]
                
                all_outputs.extend(decoded)
            
            # Format outputs to match expected structure
            for idx, output in enumerate(all_outputs):
                if idx < len(raw_datas):
                    raw_datas[idx]["prediction"] = [output]
        
        # Save results
        save_path = os.path.join(args.outdir, args.base_model.split('/')[-1]+'_'+filename.split('.')[0]+'.jsonl')
        
        with open(save_path, 'w') as f:
            for item in raw_datas:
                f.write(json.dumps(item)+'\n')


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

    args = parser.parse_args()
    
    # Adjust model_max_length based on model type
    if 'deepseek-coder-v2' in args.base_model.lower():
        # DeepSeek-Coder-V2 supports up to 128K context window
        args.model_max_length = min(args.model_max_length, 8192)  # Using 8K for practical purposes
    elif 'phi-2' in args.base_model.lower():
        args.model_max_length = min(args.model_max_length, 2048)

    run(args)
