from setuptools import find_packages, setup
from typing import List

TRIGGER = "-e ."

def get_reuirements(file_path : str) -> List[str]:
    """
    Function to Return list of libraries to install in platform.
    """
    requirements = []
    with open(file_path,"r") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if TRIGGER in requirements:
            requirements.remove(TRIGGER)
        
    return requirements


setup(
    name="MINST - Handwritten Number prediction",
    version="0.0.1",
    author="Abhijit Darekar",
    author_email="abhijit.darekar001@gmail.com",
    install_requires=get_reuirements('requirements.txt'),
    packages=find_packages()
)