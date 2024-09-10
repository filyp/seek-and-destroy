import torch as pt
from torcheval.metrics.text import Perplexity

device = "cuda"


def dataset_to_equal_chunks(dataset, tokenizer, chunk_size=100):
    # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    uneven_tokens = tokenizer(dataset["text"])["input_ids"]
    chunks = []
    for ts in uneven_tokens:
        for i in range(0, len(ts) - chunk_size, chunk_size):
            chunks.append(ts[i : i + chunk_size])
    return pt.tensor(chunks)


def eval_perplexity(model, chunks, batch_size=32):
    metric = Perplexity(device=device)
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size].to(device)
        with pt.no_grad():
            outputs = model(input_ids=batch)
        metric.update(outputs.logits[:, :-1], batch[:, 1:])

        # clean up model cache and state and memory
        del outputs
        pt.cuda.empty_cache()

    return metric.compute()
