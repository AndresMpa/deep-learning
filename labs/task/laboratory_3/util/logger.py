import subprocess


def expected_time(msg):
    try:
        subprocess.run(f"figlet {msg} | lolcat",
                       shell=True, stdout=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print(f"{msg}")
    except subprocess.CalledProcessError as error:
        print(f"Unexpected error: {error}")
