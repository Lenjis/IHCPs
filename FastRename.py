import os
import re

pattern = re.compile(r"[0-9]*(?=A_)")


def RemoveIndex(path=None):
    """
    Parameters:
    path (str, optional): The directory path to be processed. Defaults to the current directory.
    """
    if path is None:
        dir = "./"
    else:
        dir = path
    print("Removing index...")
    try:
        for filename in os.listdir(dir):
            match = re.search(pattern, filename)
            if match:
                newname = filename.replace(match.group(), "")
                os.rename(os.path.join(dir, filename), os.path.join(dir, newname))
        print("All indexes removed.")
    except Exception as e:
        print(f"An error occurred: {e}")


def AddIndex(path=None, start_idx=1, end_idx=None):
    """
    Parameters:
    start_idx (int, optional): The starting index. Defaults to 1.
    end_idx (int, optional): The ending index. If not provided, indexes will be added to all files.
    """
    idx = start_idx
    if path is None:
        dir = "./"
    else:
        dir = path
    print("Adding index...")
    try:
        if end_idx == None:
            for filename in os.listdir(dir):
                match = re.search(pattern, filename)
                if match:
                    newname = str(idx) + filename
                    os.rename(os.path.join(dir, filename), os.path.join(dir, newname))
                    idx += 1
        else:
            for filename in os.listdir(dir):
                if idx <= end_idx:
                    match = re.search(pattern, filename)
                    if match:
                        newname = str(idx) + filename
                        os.rename(os.path.join(dir, filename), os.path.join(dir, newname))
                        idx += 1
        print("Index added.")
    except Exception as e:
        print(f"An error occurred: {e}")
