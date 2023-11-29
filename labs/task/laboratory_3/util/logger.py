import subprocess


def expected_time(msg):
    try:
        subprocess.run(f"figlet {msg} | lolcat", shell=True)
    except FileNotFoundError:
        print(f"{msg}")
    except subprocess.CalledProcessError as error:
        print(f"Unexpected error: {error}")
