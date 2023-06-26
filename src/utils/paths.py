from pathlib import Path

def get_project_path() -> str:
    return Path(__file__).parent.parent.parent

if __name__ == "__main__":
    print(get_project_path())