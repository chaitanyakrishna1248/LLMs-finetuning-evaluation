# LLMs-finetuning-evaluation

This repo is part of my existing project with Amazon, LLMs for Edge devices. The repo consists of fine-tuning and evaluation scripts for running various state-of-the-art models like LLaMA2, LLaMA3, Gemma and evaluating on open assistant and medico datasets. The scripts can be run on the command line with just few arguments including model id, dataset, LoRA parameters etc. The repo is still under development.

Example CLI command: python finetune_lora.py --model_id meta-llama/Llama-2-7b-chat-hf --dataset_path /home/asdw/amazon-llm/code/CKM/dataset/medquad_instruct_train_12k.json --output_dir /media/asdw/130a808e-ec26-42a1-93ab-42b857d97bd4/capstone_files/llama2-7b-qa-med-16k-max-len-512-r64-a16 --lora_alpha 16 --lora_dropout 0.1 --r 64 --num_train_epochs 3 --max_seq_length 512
