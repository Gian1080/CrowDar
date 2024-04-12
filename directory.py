import os
import pkg_resources

def create_unique_result_directory(epoch, unique_id):
    """
    Creates a unique directory for storing results of a specific model training epoch.
    
    Parameters:
        epoch (int): The epoch number for which the directory is being created.
        unique_id (str): A unique identifier, typically a date-time string, to distinguish directories.

    Returns:
        str: The path to the newly created unique directory.
    """
    # Base directory for results
    base_dir = 'Results'
    
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create a unique directory name using the date-time and epoch
    unique_dir_name = f"{base_dir}/{unique_id}/_epoch_{epoch}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(unique_dir_name):
        os.makedirs(unique_dir_name)
    
    return unique_dir_name

def dependency_versions():
    """
    Retrieves the versions of all installed packages in the current environment.
    
    Returns:
        dict: A dictionary where keys are package names and values are their respective versions.
    """
    dependencies = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    return dependencies
