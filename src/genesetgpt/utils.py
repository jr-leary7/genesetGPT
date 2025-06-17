import numpy as np

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

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two `numpy` arrays. 

    Parameters
    ----------
    a : np.ndarray
        A `numpy` array. 
    b : np.ndarray
        A `numpy` array of the same dimension as `a`. 

    Returns
    -------
    res : float 
        A float specifying the cosine similarity between `a` and `b`. 
    """
    res = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return res 
