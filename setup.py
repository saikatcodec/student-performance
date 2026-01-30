from setuptools import setup, find_packages

def get_requirements(file_path: str) -> list[str]:
    requirements = []
    with open(file_path, 'r') as file:
        requirements = [line.strip() for line in file.readlines()]

    return requirements

setup(
    name='student-performance',
    version='1.0.0',
    author='Joy Kumar Acharjee',
    author_email='joy29940@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)