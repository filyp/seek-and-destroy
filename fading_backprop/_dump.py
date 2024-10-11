# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"

# dataset = load_dataset("open_subtitles", lang1="en", lang2="pl", trust_remote_code=True)
# dataset = load_dataset("oscar-corpus/oscar", "unshuffled_deduplicated_pl", trust_remote_code=True)

# load_dataset(
#     "oscar-corpus/OSCAR-2301",  # big multilingual corpus
#     "pl",
#     trust_remote_code=True,
#     streaming=True,  # stream because it's huge
#     split="train",  # train is the only split in OSCAR-2301
# )


    # .batch(batch_size=batch_size, drop_last_batch=True)
    # # consolidate the batches from a list into a tensor
    # .map(
    #     lambda batch: dict(
    #         input_ids=pt.cat(batch["input_ids"]),
    #         attention_mask=pt.cat(batch["attention_mask"]),
    #     )
    # )