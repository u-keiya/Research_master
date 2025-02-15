from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def add_model_output(model, tokenizer, sample):
    instruction = sample['instruction']
    input = sample['input']
    query = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n" + instruction + '\n' + input + "<|end|>\n<|assistant|>\n"
    query_tensor = tokenizer(query, return_tensors="pt").input_ids
    output = model.generate(query_tensor.to('cuda'), max_length=1024, pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
    sample["reference_output"] = tokenizer.decode(output.sequences[0][len(query_tensor.squeeze()):], skip_special_tokens=True)
    scores = []
    for s in output.scores:
        scores.append(s.to('cpu'))
    sample["logits"] = scores
    return sample

def main():
    num_data = 8192
    data = load_dataset("tatsu-lab/alpaca", split='train')
    data = data.select([i for i in range(num_data)])

    path = "/path/to/model"
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    data = data.map(lambda sample: add_model_output(model, tokenizer, sample))

    data.save_to_disk('my_data/alpaca_model_logits')

if __name__=="__main__":
    main()