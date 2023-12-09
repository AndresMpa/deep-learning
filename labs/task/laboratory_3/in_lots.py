import subprocess
import platform

from lots.training import training_combinations


def update_env_parameters(parameters):
    """
    Updates .env file with the corresponding parameters

    Args:
        - parameters (Dict): A dictionary of permutations for .env file
    """
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


def init_env():
    """
    Initialize a virtual environment depending on current OS
    """
    os_name = platform.system()

    if os_name == "Windows":
        activate_cmd = "/env/Scripts/activate"
        subprocess.call(activate_cmd, shell=True)
    else:
        activate_cmd = "source env/bin/activate"
        subprocess.call(activate_cmd, shell=True)


init_env()

parameter_combinations = training_combinations
for parameters in parameter_combinations:
    update_env_parameters(parameters)

    subprocess.call(["python", "main.py"], shell=True)
