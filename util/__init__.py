def print_if(msg: str, cond: bool):
    if cond:
        print(msg)


def is_interactive(name: str) -> bool:
    return name == '__main__'


def print_if_interactive(msg: str, name: str):
    print_if(msg, is_interactive(name))

