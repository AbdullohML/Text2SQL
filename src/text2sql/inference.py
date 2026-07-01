import torch


def generate_sql(model, tokenizer, input_text: str, max_target_length: int, num_beams: int = 4) -> str:
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=num_beams,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_many(model, tokenizer, input_texts: list[str], max_input_length: int, max_target_length: int, batch_size: int = 16):
    predictions = []

    for start in range(0, len(input_texts), batch_size):
        batch = input_texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=4,
            )

        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    return predictions
