import os


def log_training_conditions(
    args, save_dir: str, name: str="params.txt",
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
