def add_trailing_period(text: str) -> str:
    """
    Add a final period to a string. 

    Parameters
    ----------
    text : str
        A string to be edited. Defaults to None. 

    Returns
    -------
    text : str
        A string with a trailing period added if needed. 
    """
    if not text.endswith('.'):
        return text + '.'
    return text
