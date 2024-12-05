import os
import re
import shutil

pattern = re.compile(r"[0-9]*(?=A_)")


def Merge(path1: str, path2: str):
    """
    Parameters:
    path (str, optional): The directory path to be processed.
    """
    print("Merging files...")
    try:
        for filename in os.listdir(os.path.join(path1, "input")):
            shutil.move(
                os.path.join(path1, "input", filename), os.path.join(path2, "input")
            )
        for filename in os.listdir(os.path.join(path1, "output")):
            shutil.move(
                os.path.join(path1, "output", filename), os.path.join(path2, "output")
            )
        print("Files merged.")
    except Exception as e:
        print(f"An error occurred: {e}")


def RemoveIndex(path="./"):
    """
    Parameters:
    path (str, optional): The directory path to be processed. Defaults to the current directory.
    """
    print("Removing index...")
    try:
        for filename in os.listdir(path):
            match = re.search(pattern, filename)
            if match:
                newname = filename.replace(match.group(), "", 1)
                os.rename(os.path.join(path, filename), os.path.join(path, newname))
        print("All indexes removed.")
    except Exception as e:
        print(f"An error occurred: {e}")


def AddIndex(path="./", start_idx=1, end_idx=None):
    """
    Parameters:
    start_idx (int, optional): The starting index. Defaults to 1.
    end_idx (int, optional): The ending index. If not provided, indexes will be added to all files.
    """
    idx = start_idx
    print("Adding index...")
    try:
        if end_idx == None:
            for filename in os.listdir(path):
                match = re.search(pattern, filename)
                if match:
                    newname = str(idx) + filename
                    os.rename(os.path.join(path, filename), os.path.join(path, newname))
                    idx += 1
        else:
            for filename in os.listdir(path):
                if idx <= end_idx:
                    match = re.search(pattern, filename)
                    if match:
                        newname = str(idx) + filename
                        os.rename(
                            os.path.join(path, filename), os.path.join(path, newname)
                        )
                        idx += 1
        print("Index added.")
    except Exception as e:
        print(f"An error occurred: {e}")


def Rename(path1: str, path2: str, start_idx=1, end_idx=None):
    '''
    Parameters:
    path1 (str): The directory path to be processed.
    path2 (str): The directory path to be processed.
    start_idx (int, optional): The starting index. Defaults to 1.
    end_idx (int, optional): The ending index. If not provided, indexes will be added to all files.
    '''
    Merge(path1, path2)
    RemoveIndex(os.path.join(path2, "input"))
    RemoveIndex(os.path.join(path2, "output"))
    AddIndex(os.path.join(path2, "input"), start_idx, end_idx)
    AddIndex(os.path.join(path2, "output"), start_idx, end_idx)
