import subprocess
import platform


def expected_time(msg):
    try:
        if platform.system() == "Windows":
            print(msg)
        else:
            subprocess.run(f"figlet {msg} | lolcat",
                           shell=True, stdout=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print(f"{msg}")
    except subprocess.CalledProcessError as error:
        print(f"Unexpected error: {error}")
