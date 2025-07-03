import pandas as pd

def display_dataframe_to_user(name: str, dataframe: pd.DataFrame) -> None:
    """Fallback display function for notebooks and scripts."""
    header = f"=== {name} ===" if name else ""
    if header:
        print(header)
    try:
        from IPython.display import display
        display(dataframe)
    except Exception:
        print(dataframe)
