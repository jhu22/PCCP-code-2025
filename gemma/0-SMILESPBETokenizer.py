import smiles_llm as llm
filename='data/psk.txt'

checkpoint = "checkpoints/psk"

tokenizer = llm.SMILESBPETokenizer(dropout=None)

alphabet = list(llm.SMILESAlphabet().get_alphabet())

hyperparams = {"batch_size": 256, "max_epochs": 30, "min_epochs": 15,
               "max_length": 64, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 12, "n_embd": 12 * 48}

tokenizer.train(filename,vocab_size=hyperparams["vocab_size"] + len(alphabet),min_frequency=hyperparams["min_frequency"],initial_alphabet=alphabet)


tokenizer.save_model(checkpoint)
tokenizer.save(f"{checkpoint}/tokenizer.json")
