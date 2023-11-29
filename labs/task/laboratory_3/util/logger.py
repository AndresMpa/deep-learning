import subprocess


def expected_time(msg):
    try:
        subprocess.run(["figlet", msg, "| lolcat"])
    except FileNotFoundError:
        print(f"{msg}")
    except subprocess.CalledProcessError as error:
        print(f"Unexpected error: {error}")
