import subprocess


def update_env_parameters(parameters):
    # Read the content of the .env file
    env_file_path = ".env"
    with open(env_file_path, 'r') as file:
        lines = file.readlines()

    # Iterate through each line and update the target parameters if found
    for i in range(len(lines)):
        for target_parameter, new_value in parameters.items():
            if lines[i].startswith(f"{target_parameter}="):
                lines[i] = f"{target_parameter}={new_value}\n"

    # Write the updated content back to the .env file
    with open(env_file_path, 'w') as file:
        file.writelines(lines)


parameter_combinations = [
    {"NET_ARCH": "AlexNet", "ITERATIONS": "1", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "AlexNet", "ITERATIONS": "5", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "AlexNet", "ITERATIONS": "10", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "AlexNet", "ITERATIONS": "1", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "AlexNet", "ITERATIONS": "5", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "AlexNet", "ITERATIONS": "10", "LEARNING_RATE": "0.03"},

    {"NET_ARCH": "VGG16", "ITERATIONS": "1", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "VGG16", "ITERATIONS": "5", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "VGG16", "ITERATIONS": "10", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "VGG16", "ITERATIONS": "1", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "VGG16", "ITERATIONS": "5", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "VGG16", "ITERATIONS": "10", "LEARNING_RATE": "0.03"},

    {"NET_ARCH": "VGG19", "ITERATIONS": "1", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "VGG19", "ITERATIONS": "5", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "VGG19", "ITERATIONS": "10", "LEARNING_RATE": "0.01"},
    {"NET_ARCH": "VGG19", "ITERATIONS": "1", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "VGG19", "ITERATIONS": "5", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "VGG19", "ITERATIONS": "10", "LEARNING_RATE": "0.03"},

    {"NET_ARCH": "AlexNet", "ITERATIONS": "100", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "VGG16", "ITERATIONS": "100", "LEARNING_RATE": "0.03"},
    {"NET_ARCH": "VGG19", "ITERATIONS": "100", "LEARNING_RATE": "0.03"},
]

activate_cmd = "/env/Scripts/activate"
subprocess.call(activate_cmd, shell=True)

# Update parameters for each combination and call main.py
for parameters in parameter_combinations:
    update_env_parameters(parameters)

    subprocess.call(["python", "main.py"], shell=True)
