import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pickle

def format_query(sample):
    instruction = sample['instruction']
    input = sample['input']
    query = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n" + instruction + '\n' + input + "<|end|>\n<|assistant|>\n"
    sample['query'] = query
    return sample

def make_reference(model, data, pad_id):
    sequence_ref = {}
    logits_ref = {}
    for d in tqdm(data):
        query_tensor = torch.tensor(d['input_ids']).to("cuda").unsqueeze(0)
        outputs = model.generate(query_tensor, max_new_tokens=1024, pad_token_id=pad_id, output_scores=True, return_dict_in_generate=True)
        sequence_ref[d['id']] = outputs.sequences[0].to("cpu")
        scores = []
        for s in outputs.scores:
            scores.append(s.to('cpu'))
        logits_ref[d['id']] = scores
    torch.cuda.empty_cache()
    return (sequence_ref, logits_ref)

def main():
    # データセットの準備
    n_data = 1024
    data = load_dataset("tatsu-lab/alpaca", split='train')
    data = data.add_column('id', [i for i in range(len(data))])
    path = "/path/to/model"
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")

    v = 1   # 4096個のデータを一気に作れないので分割

    start = n_data * (v-1)
    end = n_data * v
    data_v = data.select([i for i in range(start, end)])
    data_v = data_v.map(format_query, remove_columns=['instruction', 'input', 'text'])
    data_v = data_v.map(lambda sample: tokenizer(sample['query']))
    outputs_ref = make_reference(model=model, data=data_v, pad_id=tokenizer.eos_token_id)
    with open(f'reference_data/alpaca_{n_data}_{v}_modelname.pickle', mode='wb') as f:
        pickle.dump(outputs_ref, f)

if __name__=="__main__":
    main()