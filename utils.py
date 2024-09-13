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


############################################
# harmful_eval.py helpers


def immediately_remove_param_gradients_to_save_memory(model):
    # register a hook on each module that deletes parameter grad after backprop
    def cleanup_param_grad_hook(module, grad_input, grad_output):
        for param in module.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = None
                pt.cuda.empty_cache()

    for name, module in model.named_modules():
        module.register_full_backward_hook(cleanup_param_grad_hook)


def get_response_log_prob(example, model, tokenizer):
    # prepare input
    prompt = tokenizer.encode(example["prompt"], return_tensors="pt").to(device)
    response = tokenizer.encode(example["response"], return_tensors="pt").to(device)
    # note: this doesn't support batching
    assert prompt.shape[0] == response.shape[0] == 1
    # trim BOS token from response
    response = response[:, 1:]

    prompt_len = prompt.shape[1]
    full_input = pt.cat([prompt, response], dim=1)
    assert pt.all(full_input[:, prompt_len:] == response)

    # forward pass
    output = model(full_input)
    # take only the part that tries to predict the response
    pred = output.logits[:, prompt_len - 1 : -1, :]
    # gather response token probs
    all_probs = pt.softmax(pred, dim=-1)
    response_token_probs = pt.gather(all_probs, -1, response.unsqueeze(-1)).squeeze(-1)
    # ignore the first token,
    # because the model cannot know how the answer start is indicated here
    response_token_probs = response_token_probs[:, 1:]
    # log of multiplied probabilities == sum of log'd probabilities
    response_log_prob = response_token_probs.log().sum(dim=-1)
    response_len = response.shape[1]
    return response_log_prob, response_len
