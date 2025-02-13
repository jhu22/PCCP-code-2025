import smiles_llm as llm
import os
import os
import multiprocessing
from peft import LoraConfig,get_peft_model
# 先使用1epoch
hyperparams = {"batch_size": 8, "max_epochs":1, "min_epochs": 1,
               "max_length": 64, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 8, "n_embd": 12 * 48}

num_workers = 4  # Number of dataloader worker processes.

tokenizer = llm.SMILESBPETokenizer(dropout=None)

tokenizer = llm.SMILESBPETokenizer.get_hf_tokenizer('./checkpoints/psk/tokenizer.json', model_max_length=hyperparams["max_length"])

from transformers import GemmaConfig, GemmaForCausalLM
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


data_dir = "./data/psk.txt"
base_checkpoint_path = "./checkpoints/psk"



def train_model():
    
    filename = data_dir
    checkpoint_path = base_checkpoint_path
    checkpoint_path_prv = "./checkpoints/1m_4"
    model = GemmaForCausalLM.from_pretrained(f"{checkpoint_path_prv}/model", output_attentions=True)

   




    # Trainer 和 Callbacks
    checkpoint_cb = ModelCheckpoint(f"{checkpoint_path}/model/")
    early_stopping_ppl = EarlyStopping(
        monitor="ppl_epoch", patience=4, min_delta=5e-3, check_finite=True,
        stopping_threshold=1.1, divergence_threshold=hyperparams["vocab_size"] / 10,
        verbose=True, mode="min", check_on_train_epoch_end=True
    )
    
    trainer = Trainer(
        strategy="ddp", accelerator="gpu", devices=1,
        callbacks=[checkpoint_cb, early_stopping_ppl],
        max_epochs=hyperparams["max_epochs"], min_epochs=hyperparams["min_epochs"],
    )
    
    # Data loading and training
    # 适当修改数据加载的部分以适应您的框架
    lit_model = llm.LLMLitModel(
    model,
    batch_size=hyperparams["batch_size"],
    learning_rate=hyperparams["learning_rate"],
    final_learning_rate=hyperparams["final_learning_rate"],
    weight_decay=hyperparams["weight_decay"],
    adam_eps=hyperparams["adam_eps"],
    adam_betas=hyperparams["adam_betas"],
    scheduler_T_max=hyperparams["scheduler_T_max"],
    )

    datamodule = llm.LMDataModule(filename, tokenizer, batch_size=hyperparams["batch_size"], num_workers=num_workers)

    trainer.fit(lit_model, datamodule)


    # 保存模型
    model.save_pretrained(f"{checkpoint_path}/model/")




if __name__ == "__main__":
    train_model()

# nohup python 1-gemma-lora-ft.py > 1_2run.log 2>&1 &

