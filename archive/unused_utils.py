def retrain_and_eval(
    model, initial_stats, forget, retain, num_batches=10, b_size=16, lr=0.0003
):
    """takes a model after an intervention and evals it before and after retraining"""
    pre_retrain_diff = get_stats(model, forget, retain) - initial_stats
    print_stats(pre_retrain_diff)

    f_r = zip(forget["relearn"].batch(b_size), retain["relearn"].batch(b_size))
    for forget_batch, retain_batch in islice(f_r, num_batches):
        normal_train_step(model, forget_batch, lr)
        normal_train_step(model, retain_batch, lr)

    post_retrain_diff = get_stats(model, forget, retain) - initial_stats
    print_stats(post_retrain_diff)
    return pre_retrain_diff, post_retrain_diff


def scale_perturbation(model, original_state_dict, scaling_factor):
    for module_name, current_weights in model.state_dict().items():
        original_weights = original_state_dict[module_name]
        # modification need to be done in-place so it's a bit awkward:
        current_weights -= original_weights
        current_weights *= scaling_factor
        current_weights += original_weights
