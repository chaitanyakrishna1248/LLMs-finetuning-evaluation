import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model
from trl import SFTTrainer

class ModelTrainer:
    def __init__(self, model_id, dataset_path, output_dir, lora_alpha, lora_dropout, r, num_train_epochs, max_seq_length):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.r = r
        self.num_train_epochs = num_train_epochs
        self.max_seq_length = max_seq_length

        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None

    def load_dataset(self):
        self.dataset = load_dataset("json", data_files=self.dataset_path, split="train")

    def configure_model(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def configure_lora(self):
        self.peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias="none",
            target_modules=['o_proj','k_proj','q_proj','gate_proj','down_proj','v_proj', 'up_proj'],
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, self.peft_config)

    def configure_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            auto_find_batch_size=True,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="adamw_bnb_8bit",
            save_strategy="epoch",
            learning_rate=2e-4,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            weight_decay=0.001,
            lr_scheduler_type="constant",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
        )

    def train(self):
        os.environ["WANDB_DISABLED"] = "true"
        self.trainer.train()

    def save_model(self):
        self.trainer.save_model(self.output_dir)

    def run(self):
        self.load_dataset()
        self.configure_model()
        self.configure_lora()
        self.configure_trainer()
        self.train()
        self.save_model()

def main():
    
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    parser.add_argument('--model_id', type=str, required=True, help='Model identifier from Hugging Face')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the fine-tuned model')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--r', type=int, default=64, help='LoRA rank')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length for training')

    args = parser.parse_args()

    trainer = ModelTrainer(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.r,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length
    )
    trainer.run()

if __name__ == "__main__":
    main()
