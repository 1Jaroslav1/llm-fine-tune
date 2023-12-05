import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import bitsandbytes as bnb
import evaluate
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import SFTTrainer
import numpy as np
import argparse


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(list(lora_module_names))
    return list(lora_module_names)


def main(args):
    # Set environment variables
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
    my_dataset = load_dataset(args.dataset_name, split="train")
    splitted_dataset = my_dataset.train_test_split(test_size=args.eval_set_size, shuffle=True, seed=42)

    compute_dtype = getattr(torch, 'float32')

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 50)
            print("GPU supports bfloat16")
            print("=" * 50)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map=args.device_map
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.gradient_checkpointing_enable()
    model = prepare_model_for_int8_training(model)

    # Load LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
        output_dir=args.args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.global_batch_size // args.per_device_train_batch_size,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        # group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy="steps" if args.eval_set_size > 0 else "no",
        load_best_model_at_end=True if args.eval_set_size > 0 else False,
        report_to="tensorboard",
        eval_accumulation_steps=args.eval_accumulation_steps,
        logging_dir=args.logging_dir,
        logging_strategy="steps"
    )

    def compute_metrics(eval_preds):
        metric = evaluate.load("bleu", "rouge")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Convert predictions and labels to tokens
        predictions_tokens = [tokenizer.convert_ids_to_tokens(seq) for seq in predictions]
        labels_tokens = [tokenizer.convert_ids_to_tokens(seq) for seq in labels]

        # Join tokens to form strings for predictions
        predictions_str = [' '.join(seq).replace(' ▁', ' ').strip() for seq in predictions_tokens]

        # Join tokens to form strings for references and wrap each in a list
        references_str = [[' '.join(seq).replace(' ▁', ' ').strip()] for seq in labels_tokens]

        torch.cuda.empty_cache()

        score = metric.compute(predictions=predictions_str, references=references_str)

        # Return only the overall BLEU score as a scalar
        return {"bleu": score['bleu'], "rouge": score['rouge']}

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=splitted_dataset["train"],
        eval_dataset=splitted_dataset["test"] if args.eval_set_size > 0 else None,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2024,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
        compute_metrics=compute_metrics,
        infinite=True,
    )

    trainer.train()

    trainer.model.save_pretrained(args.new_model)

    del model,
    del trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )
    model = PeftModel.from_pretrained(base_model, args.new_model)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.push_to_hub(args.new_model, use_temp_dir=False)
    tokenizer.push_to_hub(args.new_model, use_temp_dir=False)


# Command-line arguments
parser = argparse.ArgumentParser(description="Model Training Script")
# General args
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name")
parser.add_argument("--dataset_name", type=str, default="atom92/medical_healthwa_2.0", help="Dataset name")
parser.add_argument("--new_model", type=str, required=True, help="Fine-tuned model name")
# LoRA args
parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA scaling")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
# Training args
parser.add_argument("--output_dir", type=str, default="./result", help="Output directory where the model predictions and checkpoints will be stored")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--fp16", type=bool, default=False, help="Enable f16")
parser.add_argument("--bf16", type=bool, default=False, help="Enable bf16")
parser.add_argument("--global_batch_size", type=int, default=32, help="Global batch size")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="The number of samples the model needs to see until the weights get updated.")
parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per GPU for training")
parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient normal (gradient clipping)")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate (AdamW optimizer)")
parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay to apply to all layers except bias/LayerNorm weights")
parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer to use")
parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate schedule")
parser.add_argument("--max_steps", type=int, default=-1, help="Number of training steps (overrides num_train_epochs)")
parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of steps for a linear warmup (from 0 to learning rate)")
parser.add_argument("--group_by_length", type=bool, default=True, help="Group sequences into batches with same length \n Saves memory and speeds up training considerably")
parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps")
parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps")
parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Evaluation accumulation steps")
parser.add_argument("--logging_dir", type=str, default="./logs", help=" TensorBoard log directory")
parser.add_argument("--neftune_noise_alpha ", type=float, default=0.1, help=" NEFTune noise embeddings")
# SFT args
parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length to use")
parser.add_argument("--packing", type=bool, default=True, help="Pack multiple short examples in the same input sequence to increase efficiency")
parser.add_argument("--device_map", type=object, default={"": 0}, help="For default load the entire model on the GPU 0")
# Dataset args
parser.add_argument("--eval_set_size", type=float, default=0.1, help="Evaluation size")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# python train.py --new_model medical_lama_2_all
