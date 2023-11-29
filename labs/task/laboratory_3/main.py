import multiprocessing
import time

from util.architecture import create_architecture, create_optimizer, create_transform, get_loss_function, save_arch
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
    # Función para registrar las activaciones de las capas intermedias
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    def hook(module, input, output):
        global activation
        activation = output.detach()

    # Registrar ganchos (hooks) en capas intermedias
    architecture.characteristic[3].register_forward_hook(
        get_activation('conv1'))
    architecture.characteristic[6].register_forward_hook(
        get_activation('conv2'))

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
                print(f'[{epochs} {j:5d} {(iteration_lost / 200):.3f}]')
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
    save_arch(architecture)
