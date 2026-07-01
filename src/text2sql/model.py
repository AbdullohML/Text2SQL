import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer_and_model(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if config.get("use_8bit", False):
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config["model_name"],
            quantization_config=quant_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
        if torch.cuda.is_available():
            model.to("cuda")

    if config.get("use_lora", False):
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["lora_target_modules"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return tokenizer, model
