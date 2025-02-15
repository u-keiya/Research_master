import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lossfunction import CustomLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import os
import pickle
import copy
from collections import defaultdict
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def torch_fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

def format_query(sample):
    instruction = sample['instruction']
    input = sample['input']
    query = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n" + instruction + '\n' + input + "<|end|>\n<|assistant|>\n" # for phi-3-mini
    #query = '<|im_start|>user\n' + instruction + '\n' + input + '<|im_end|>\n<|im_start|>assistant\n'   # for zamba2-2.7b
    sample['query'] = query
    return sample

def my_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def padding(ls, pad_id):
    max_len = 0
    ls = copy.deepcopy(ls)
    for l in ls:
        if len(l) > max_len:
            max_len = len(l)
    
    new_ls = []
    for l in ls:
        while len(l) != max_len:
            l.insert(0, pad_id)
        new_ls.append(l)
    return torch.tensor(new_ls)

def remove_padding(tensor, pad_id):
    r_i = 0
    for token in torch.flip(tensor,dims=[0]):
        if token == pad_id:
            r_i -= 1
        else:
            break
    
    if r_i==0:
        return tensor
    else:
        return tensor[:r_i]

def prepare_reference_data(n_train_data):
    # 分割して用意したデータを結合
    with open('/path/to/reference_data/alpaca_1024_1.pickle', 'rb') as f:
        outputs_ref1 = pickle.load(f)
    with open('/path/to/reference_data/alpaca_1024_2.pickle', 'rb') as f:
        outputs_ref2 = pickle.load(f)
    with open('/path/to/reference_data/alpaca_1024_3.pickle', 'rb') as f:
        outputs_ref3 = pickle.load(f)
    with open('/path/to/reference_data/alpaca_1024_4.pickle', 'rb') as f:
        outputs_ref4 = pickle.load(f)

    sequence_ref1, logits_ref1 = outputs_ref1
    sequence_ref2, logits_ref2 = outputs_ref2
    sequence_ref3, logits_ref3 = outputs_ref3
    sequence_ref4, logits_ref4 = outputs_ref4

    # キー変換関数を指定
    def key_conversion(key):
        return str(key)  # 必要に応じて他の変換関数を適用

    # 新しい辞書の作成
    new_sequence_ref1 = {key_conversion(key): value for key, value in sequence_ref1.items() if key in range(n_train_data)}
    new_sequence_ref2 = {key_conversion(key): value for key, value in sequence_ref2.items() if key in range(n_train_data)}
    new_sequence_ref3 = {key_conversion(key): value for key, value in sequence_ref3.items() if key in range(n_train_data)}
    new_sequence_ref4 = {key_conversion(key): value for key, value in sequence_ref4.items() if key in range(n_train_data)}

    new_logits_ref1 = {key_conversion(key): value for key, value in logits_ref1.items() if key in range(n_train_data)}
    new_logits_ref2 = {key_conversion(key): value for key, value in logits_ref2.items() if key in range(n_train_data)}
    new_logits_ref3 = {key_conversion(key): value for key, value in logits_ref3.items() if key in range(n_train_data)}
    new_logits_ref4 = {key_conversion(key): value for key, value in logits_ref4.items() if key in range(n_train_data)}

    new_sequence_ref = dict(**new_sequence_ref1, **new_sequence_ref2, **new_sequence_ref3, **new_sequence_ref4)
    new_logits_ref = dict(**new_logits_ref1, **new_logits_ref2, **new_logits_ref3, **new_logits_ref4)

    return (new_sequence_ref, new_logits_ref)

def take_top_p(logit, top_p):
    probs = torch.nn.functional.softmax(logit, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort <= top_p
    top_p_prob = probs_sort[mask]
    top_p_prob.div_(top_p_prob.sum(dim=-1, keepdim=True))
    top_p_id = probs_idx[mask]
    return (top_p_id, top_p_prob)

def make_branch_sets(logits_ref, ratio=0.5, top_p=0.9):
    branch_sets = {}
    for id in logits_ref.keys():
        branch_set = {}
        logits = logits_ref[id]
        for pos in range(int(len(logits)*ratio)):
            logit = logits[pos]
            top_ps = take_top_p(logit, top_p)
            if len(top_ps[0]) > 1:
                branch_set[str(pos)] = (top_ps[0][1:], top_ps[1][1:])
        branch_sets[str(id)] = branch_set
    return branch_sets

def validate(model, val_data, pad_token_id):
    lens = []
    for val in val_data:
        response = model.generate(torch.tensor([val["input_ids"]]).to("cuda"), max_new_tokens=1024, pad_token_id=pad_token_id)
        lens.append(len(response[0][len(val["input_ids"]):]))
    return sum(lens) / len(lens)

def train(model, outputs_ref, train_data, val_data, gen_kwargs, n_epoch=1, bsz=4):
    torch_fix_seed(42)
    loss_fn = CustomLoss(alpha1=0.98, alpha2=0.01, alpha3=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    sequence_ref, logits_ref = outputs_ref
    dataloader = DataLoader(dataset=train_data, batch_size=bsz, shuffle=True, collate_fn=my_collator)
    responses = defaultdict(list)

    cnt = 1

    for epoch in range(n_epoch):
        print(f"epoch {epoch}:")
        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            model.eval()
            query_tensors = padding(batch["input_ids"], gen_kwargs["pad_token_id"]).to("cuda")  # 2d list -> 2d tensor
            with torch.no_grad():
                response_tensors = model.generate(query_tensors, **gen_kwargs)  # 2d tensor

            model.train()
            loss_all = 0
            n_loss = 0
            for query_tensor, response_tensor, id, input_ids in zip(query_tensors, response_tensors, batch["id"], batch["input_ids"]):
                response_tensor = remove_padding(response_tensor[len(query_tensor):].to("cpu"), gen_kwargs["pad_token_id"])
                response_tensor_ref = sequence_ref[str(id)][len(input_ids):]

                branching_pos = -1
                for pos in range(min(len(response_tensor), len(response_tensor_ref))):
                    if response_tensor[pos] != response_tensor_ref[pos]:
                        branching_pos = pos
                        break
                
                if branching_pos < 0:
                    continue
                
                flag = 0
                if len(response_tensor) < len(response_tensor_ref):
                    flag = 1
                else:
                    flag = -1
                
                if branching_pos != 0:
                    query_tensor = torch.cat((torch.tensor(input_ids), response_tensor[:branching_pos])).unsqueeze(0)
                else:
                    query_tensor = torch.tensor(input_ids).unsqueeze(0)
                
                logit = model.forward(query_tensor.to("cuda")).logits.squeeze()[-1]
                logit_ref = logits_ref[str(id)][branching_pos].to("cuda")

                loss = loss_fn(label=response_tensor[branching_pos].to("cuda"), logit=logit, logit_ref=logit_ref, flag=flag)

                loss_all += loss
                n_loss += 1
                responses[id].append(response_tensor)
            
            if n_loss == 0:
                continue
            loss_all = loss_all / n_loss
            loss_all.backward()
            optimizer.step()

            if cnt % 64 == 0:
                val_len = validate(model=model, val_data=val_data, pad_token_id=gen_kwargs["pad_token_id"])
                print(f"valid length: {val_len}")
            cnt += 1
            torch.cuda.empty_cache()

        dir_name = f"results/modelname"
        os.mkdir(dir_name)
        model.save_pretrained(dir_name)
    with open(os.path.join(dir_name, 'responses.pickle'), 'wb') as f:
        pickle.dump(responses, f)

def train_stable(model, outputs_ref, train_data, val_data, gen_kwargs, n_epoch=1, bsz=4, weights=[0.98,0.01,0.01]):
    torch_fix_seed(42)
    loss_fn = CustomLoss(alpha1=weights[0], alpha2=weights[1], alpha3=weights[2])
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    sequence_ref, logits_ref = outputs_ref
    dataloader = DataLoader(dataset=train_data, batch_size=bsz, shuffle=True, collate_fn=my_collator)
    responses = defaultdict(list)
    gen_kwargs["do_sample"] = False

    cnt = 1

    ratio = 0.5
    path = f"reference_data/branch_sets_{ratio}_{gen_kwargs['top_p']}_4096_pickle"

    if os.path.isfile(path):
        with open(path, 'rb') as f:
            branch_sets = pickle.load(f)
    else:
        branch_sets = make_branch_sets(logits_ref, ratio=ratio, top_p=gen_kwargs['top_p'])
        with open(path, 'wb') as f:
            pickle.dump(branch_sets, f)

    for epoch in range(n_epoch):
        print(f"epoch {epoch}:")
        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            model.eval()
            ids = []
            input_ids = []
            branching_poses = []
            for id, input_id in zip(batch["id"], batch['input_ids']):
                branch_set = branch_sets[str(id)]
                if len(branch_set) != 0:
                    branching_pos = int(random.choice(list(branch_set)))
                    top_p_id, top_p_prob = branch_set[str(branching_pos)]
                    branching_token = torch.multinomial(top_p_prob, num_samples=1)
                    branching_token = torch.gather(top_p_id, -1, branching_token)
                    ids.append(id)
                    input_ids.append(torch.cat((sequence_ref[str(id)][:len(input_id)+branching_pos], branching_token)).tolist())
                    branching_poses.append(branching_pos)
            
            query_tensors = padding(input_ids, gen_kwargs["pad_token_id"]).to("cuda")  # 2d list -> 2d tensor
            with torch.no_grad():
                response_tensors = model.generate(query_tensors, **gen_kwargs)  # 2d tensor

            model.train()
            loss_all = 0
            n_loss = 0
            for query_tensor, response_tensor, id, input_id, branching_pos in zip(query_tensors, response_tensors, ids, input_ids, branching_poses):
                len_response_tensor = len(remove_padding(response_tensor[len(query_tensor):].to("cpu"), gen_kwargs["pad_token_id"]))
                len_response_tensor_ref = len(sequence_ref[str(id)][len(input_id):])
                
                flag = 0
                if len_response_tensor < len_response_tensor_ref:
                    flag = 1
                else:
                    flag = -1

                logit = model.forward(torch.tensor(input_id)[:-1].unsqueeze(0).to("cuda")).logits.squeeze()[-1]
                logit_ref = logits_ref[str(id)][branching_pos].to("cuda")

                loss = loss_fn(label=torch.tensor(input_id)[-1].to("cuda"), logit=logit, logit_ref=logit_ref, flag=flag)

                loss_all += loss
                n_loss += 1
                responses[id].append(response_tensor)
            
            if n_loss == 0:
                continue
            loss_all = loss_all / n_loss
            loss_all.backward()
            optimizer.step()

            if cnt % 64 == 0:
                val_len = validate(model=model, val_data=val_data, pad_token_id=gen_kwargs["pad_token_id"])
                print(f"valid length: {val_len}")
            cnt += 1
            torch.cuda.empty_cache()

        dir_name = f"results/modelname"
        os.mkdir(dir_name)
        model.save_pretrained(dir_name)
    with open(os.path.join(dir_name, 'responses.pickle'), 'wb') as f:
        pickle.dump(responses, f)


def main(weights):
    # データセットの準備
    print("Loading data...")
    n_train_data = 4096
    n_val_data = 50
    data = load_dataset("tatsu-lab/alpaca", split='train')
    data = data.add_column('id', [i for i in range(len(data))])
    data = data.select([i for i in range(n_train_data+n_val_data)])
    data = data.map(format_query, remove_columns=['instruction', 'input', 'text'])

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/model")
    tokenizer.pad_token = tokenizer.unk_token   # for phi-3-mini
    data = data.map(lambda sample: tokenizer(sample['query']))

    train_data = data.select([i for i in range(n_train_data)])
    val_data = data.select([i for i in range(n_train_data, n_train_data+n_val_data)])

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("/path/to/model", device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")
    
    print("Loading reference data...")
    outputs_ref = prepare_reference_data(n_train_data)

    for p in model.parameters():
        p.requires_grad = False
    
    for name, p in model.named_parameters():
        if "qkv_proj" in name:  # for phi-3-mini
        # if "in_proj" in name: # for zamba2-2.7b
            p.requires_grad = True
    
    gen_kwargs = {
        "min_new_tokens": 2,
        #"min_new_tokens": 1,   # for zamba2-2.7b
        "top_p": 0.9,
        "top_k": 0,
        "max_new_tokens": 1024,
        "do_sample": True,
        "pad_token_id": tokenizer.unk_token_id
        }

    print("training...")
    # 分岐を言語モデルに任せる方法
    #train(model=model, outputs_ref=outputs_ref, train_data=train_data, val_data=val_data, gen_kwargs=gen_kwargs, n_epoch=1, bsz=8)
    
    # 分岐をサンプリングによって実現する方法
    train_stable(model=model, outputs_ref=outputs_ref, train_data=train_data, val_data=val_data, gen_kwargs=gen_kwargs, n_epoch=1, bsz=8, weights=weights)

if __name__=="__main__":
    main(weights=[0.98, 0.01, 0.01])