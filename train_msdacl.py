from baselines.MSDA_CL.MSDACLTrainer import MSDACLTrainer

if __name__ == "__main__":
    # Pseudo-label test
    from options.train_options import MSDACLTrainOptions
    args = MSDACLTrainOptions().parse()

    msdacl = MSDACLTrainer(args)

    msdacl.msdacl_main()
