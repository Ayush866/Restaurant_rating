from setuptools import find_packages,setup
from typing import List

REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    requirements = []
    with open(REQUIREMENT_FILE_NAME) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements
setup(
    name = "restaurant_rating",
    version = "0.0.1",
    author = "Ayush",
    author_email="ayush786bisht@gmail.com",
    packages = find_packages(),
    install_requirements = get_requirements()


)