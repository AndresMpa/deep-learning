from config.vars import env_vars


def normalize_activation(activation):
    """
    normalize_activation Normalize an activation into a numpy array

    :param activation: Should be hook in a layer like activations["conv1"]
    """
    # Dims to avoid excess glare
    activation = activation[env_vars.img_start_index, :, :, :].cpu().numpy()

    numerator = activation - activation.min()
    denominator = activation.max() - activation.min()

    activation_normalized = numerator / denominator

    return activation_normalized
