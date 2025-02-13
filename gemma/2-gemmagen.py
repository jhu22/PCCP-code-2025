import torch
import numpy as np
import smiles_llm as llm

checkpoint_path_prv = "./checkpoints/psk"
tokenizer = llm.SMILESBPETokenizer(dropout=None)

hyperparams = {"batch_size": 1, "max_epochs": 30, "min_epochs": 5,
               "max_length": 64, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 8, "n_embd": 12 * 48}

tokenizer = llm.SMILESBPETokenizer.get_hf_tokenizer('./checkpoints/TRPV4_ft/tokenizer.json', model_max_length=hyperparams["max_length"])


from transformers import GemmaConfig, GemmaForCausalLM

model = GemmaForCausalLM.from_pretrained(f"{checkpoint_path_prv}/model", output_attentions=True)


import tqdm
model.eval()

generated_smiles_list = []
n_generated = 5000

for _ in tqdm.tqdm(range(n_generated)):
    # Generate from "<s>" so that the next token is arbitrary.
    smiles_start = torch.LongTensor([[tokenizer.bos_token_id]])
    # Get generated token IDs.
    generated_ids = model.generate(smiles_start,
                                   max_length=hyperparams["max_length"],
                                   do_sample=True, top_p=hyperparams["top_p"],
                                   pad_token_id=tokenizer.eos_token_id)
    # Decode the IDs into tokens and remove "<s>" and "</s>".
    generated_smiles = tokenizer.decode(generated_ids[0],
                                        skip_special_tokens=True)
    generated_smiles_list.append(generated_smiles)

np.save('./gen_mols/psk_gen_gemma_5k.npy', generated_smiles_list)
# nohup python 2-gemmagen.py > 2_2run.log 2>&1 &

