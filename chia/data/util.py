def batches_from(samples, batch_size):
    complete_batches = []
    current_batch = []
    for sample in samples:
        current_batch.append(sample)
        if len(current_batch) == batch_size:
            complete_batches.append(current_batch)
            current_batch = []

    if len(current_batch) > 0:
        complete_batches.append(current_batch)

    return complete_batches