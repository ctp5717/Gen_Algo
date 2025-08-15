import warnings

def suppress_third_party_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
        module="pandas_ta",
    )
