from transformers import AutoTokenizer
import trl
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from reward_model import RewardModel
from datasets import load_from_disk

import torch
from tqdm import tqdm
from collections import defaultdict
import pickle
import os

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    trl.set_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ppo_config = PPOConfig(
    model_name="/path/to/model",
    learning_rate=5e-6,
    ppo_epochs=4,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    adap_kl_ctrl=False,
    init_kl_coef=0.01,
    remove_unused_columns=False,
    kl_penalty="full"
    )

def format_query(sample):
    instruction = sample['instruction']
    input = sample['input']
    query = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n" + instruction + '\n' + input + "<|end|>\n<|assistant|>\n"
    sample['query'] = query
    return sample

def my_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(lam):
    set_seed(42)
    num_epoch = 1
    num_data = 4096
    n_val_data = 50
    data = load_from_disk('/path/to/data')
    data = data.add_column('id', [i for i in range(len(data))])
    data = data.select([i for i in range(num_data+n_val_data)])
    data = data.map(format_query, remove_columns=['instruction', 'input', 'text'])

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = tokenizer.unk_token   # for phi-3-mini
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'
    data = data.map(lambda sample: tokenizer(sample['query']))

    train_data = data.select([i for i in range(num_data)])
    val_data = data.select([i for i in range(num_data, num_data+n_val_data)])

    reward_model = RewardModel(tokenizer=tokenizer, lam=lam)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")

    for p in model.pretrained_model.parameters():
        p.requires_grad = False
    
    for name, p in model.pretrained_model.named_parameters():
        if "qkv_proj" in name:  # for phi-3-mini
        # if "in_proj" in name: # for zamba2-2.7b
            p.requires_grad = True

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_data,
        data_collator=my_collator
        )
    
    gen_kwargs = {
        "min_length": -1,
        # "min_length": 1, for zamba2-2.7b
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 1024,
        "do_sample": True,
        "pad_token_id": tokenizer.unk_token_id, #for phi-3-mini
        }
    
    responses = defaultdict(list)
    stats = defaultdict(list)
    
    cnt = 1

    for epoch in range(num_epoch):
        print(f"epoch {epoch}:")
        with tqdm(ppo_trainer.dataloader, total=len(ppo_trainer.dataloader), leave=False) as pbar:
            for batch in pbar:
                query_tensors = [torch.tensor(l).to("cuda") for l in batch["input_ids"]]

                # テキストを生成する
                response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **gen_kwargs)
                removed_pad_response_tensors = [r[r!=gen_kwargs["pad_token_id"]] for r in response_tensors]
                removed_pad_response_tensors = [torch.cat((r, torch.tensor([gen_kwargs["pad_token_id"]]).to('cuda'))) for r in removed_pad_response_tensors]
                batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
                batch["response_length"] = [len(tokenizer(res).input_ids) for res in batch['response']]

                for id, res in zip(batch['id'], batch['response']):
                    responses[id].append(res)

                batch["output_length"] = [len(tokenizer(out).input_ids) for out in batch['reference_output']]

                # Reward modelに入力し、rewardを受け取る
                rewards = reward_model(batch['response'], batch['reference_output'], batch["response_length"], batch['output_length'])

                # 学習ステップを実行し、metricを取得
                stat = ppo_trainer.step(query_tensors, removed_pad_response_tensors, rewards)

                for key in stat.keys():
                    stats[key].append(stat[key])
                
                if cnt % 64 == 0:
                    lens = []
                    for val in val_data:
                        response = model.generate(torch.tensor([val["input_ids"]]).to("cuda"), max_new_tokens=1024, pad_token_id=tokenizer.unk_token_id)
                        lens.append(len(response[0][len(val["input_ids"]):]))
                    print(sum(lens) / len(lens))
                cnt += 1
                torch.cuda.empty_cache()
    
    dir_name = f"results/model_name"
    os.mkdir(dir_name)
    model.save_pretrained(dir_name)
    with open(os.path.join(dir_name, 'responses.pickle'), 'wb') as f:
        pickle.dump(responses, f)
    with open(os.path.join(dir_name, 'stats.pickle'), 'wb') as f:
        pickle.dump(stats, f)

if __name__=="__main__":
    main(lam=0.3)