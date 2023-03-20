import os


def log_training_conditions(
    args, save_dir: str, name: str = "params.txt",
):
    """Save the training parameters

    Parameters
    ----------
    args: 
        Arguments
    save_dir: `str`
        Directory name
    name: `str`
        File name
    """
    with open(os.path.join(save_dir, name), 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            f.write("%s: %s\n" % (arg, val))


def log_metrics(
    metrics: dict,
    epoch: int,
    save_dir: str,
    name: str = "metrics.txt",
    write_header: bool = False,
    convert_to_percentage: bool = True,
    delimiter: str = '&',
) -> None:
    """Save the training parameters

    Parameters
    ----------
    metrics: `dict`
        Arguments
    epoch: `int`
        Epoch number
    save_dir: `str`
        Directory name
    name: `str`
        File name
    write_header: `bool`
        `True` to write the labels of the metrics
    convert_to_percentage: `bool`
        Multiply the values by 100 to make it percentage
    delimiter: `str`
        Character/string to separate the values in the text
    """
    if convert_to_percentage:
        for k in metrics.keys():
            metrics[k] *= 100
    with open(os.path.join(save_dir, name), 'a') as f:
        if write_header:
            is_first = True
            for k in metrics.keys():
                if is_first:
                    f.write("%s" % (k))
                    is_first = False
                else:
                    f.write("%s%s" % (delimiter, k))

            f.write("\n")

        is_first = True
        for k in metrics.keys():
            if is_first:
                f.write("%s" % (str(metrics[k])))
                is_first = False
            else:
                f.write("%s%s" % (delimiter, str(metrics[k])))

        f.write("\n")
