import time
import multiprocessing

from util.architecture import create_architecture, create_optimizer
from util.architecture import create_transform, get_loss_function, save_arch
from util.activation import get_activation_hook, global_activation_hook
from util.activation import normalize_activation
from util.draw import draw_views, draw_error
from util.dataset import create_dataset

from config.vars import env_vars

if __name__ == '__main__':
    start_time = time.time()

    """
    Threads management
    """
    multiprocessing.freeze_support()

    """
    Architecture definition
    """
    architecture, device = create_architecture()
    transform = create_transform()

    """
    Hyper parameters
    """
    lost_criteria = get_loss_function()
    optimizator = create_optimizer(architecture)

    """
    Analytics
    """
    error = []

    """
    Dataset split
    """
    trainloader, testloader = create_dataset(transform)

    """
    Hooks
    """
    activations = {}

    # Mapping between layer indices and names
    layer_mapping = {
        3: 'conv1',
        6: 'conv2',
    }

    # Register hooks based on the mapping
    for layer_index, layer_name in layer_mapping.items():
        hook_function = get_activation_hook(layer_name, activations)
        architecture.characteristic[layer_index].register_forward_hook(
            hook_function)

    # Register a global hook
    architecture.register_forward_hook(global_activation_hook)

    '''
    Trainig process
    '''
    for epochs in range(env_vars.iterations):
        iteration_lost = 0
        for j, i in enumerate(trainloader, 0):
            X, Y = i
            X, Y = X.to(device), Y.to(device)

            optimizator.zero_grad()
            output = architecture(X)

            lost = lost_criteria(output, Y)
            lost.backward()

            optimizator.step()
            iteration_lost += lost

            if (j % 200 == 0):
                print(f'Epoch: {epochs+1} \t \
                      Iteration: {j:5d} \t \
                      Lost: {(iteration_lost/200):.3f}')
                error.append(iteration_lost)
                iteration_lost = 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Took: {elapsed_time / 60.0:0.2f} minutes (Processing on GPU)")

    # Move the model back to the CPU
    architecture.to("cpu")

    '''
    Visualization
    '''
    activation_normalized = normalize_activation(activations["conv1"])

    '''
    Plotting activations
    '''
    timestamp = time.time()
    draw_views(activation_normalized, "conv1", timestamp)

    '''
    Plotting loss function
    '''
    # Move the error to the CPU before plotting
    error_cpu = [e.cpu().item() for e in error]
    draw_error(error_cpu, timestamp)

    '''
    Saving model
    '''
    save_arch(architecture, timestamp)