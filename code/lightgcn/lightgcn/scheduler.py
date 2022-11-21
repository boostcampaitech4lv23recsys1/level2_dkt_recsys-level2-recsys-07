from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_scheduler(optimizer):
    scheduler = ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, mode="max", verbose=True
    )
    return scheduler