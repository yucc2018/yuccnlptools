
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def genernal_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == 'smp-rank':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
