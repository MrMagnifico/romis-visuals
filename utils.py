from os import makedirs, path

def normalise(raw: list[float]) -> list[float]:
    raw_sum = sum(raw)
    return [i / raw_sum for i in raw]

def create_if_not_exists(directory_path):
    if not path.exists(directory_path):
        makedirs(directory_path)
