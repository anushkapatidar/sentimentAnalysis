
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements.txt file and returns the list of dependencies.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="sentiment_analysis_project",
    version="0.1.0",
    author="Anushka",
    author_email="patidaranushka.ap@gmail.com",
    description="A sentiment analysis Project",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)