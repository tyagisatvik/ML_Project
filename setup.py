from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        requirements = [
            line.strip()
            for line in file
            if line.strip() and not line.startswith('#')
        ]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Satvik',
    author_email='satviktyagi3@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)


    







