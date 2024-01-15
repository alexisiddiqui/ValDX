# code to generate the path variable for pytest
import os

def env_var_to_string(var_name):
    """Get the value of an environment variable as a string"""
    try:
        return os.environ[var_name]
    except KeyError:
        print(f"Please check the environment variable {var_name}")
        return None
    
# now we can use this function to generate the path variable for pytest
# this code will be run as a script in main so that we can get the environment variable