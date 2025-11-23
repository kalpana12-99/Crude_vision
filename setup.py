## this is basically for building our application as a package , and just deploy like python PyPI->Puthon package Index(check it out)
##In Python, setup.py is a module used to build and distribute Python packages. It typically contains information about the package, such as its name, version, and dependencies, as well as instructions for building and installing the package.


from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT= '-e .'


def get_requirements(file_path:str)->List[str]:   
    #this function will retunr the list of requiremts 
    requirements = []  #make a list
    with open(file_path) as file_obj: #opening the file and as a temporary file object
        requirements = file_obj.readline()
        requirements = [req.replace("\n"," ")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Crude Vision',
    version='0.0.1',
    author='kalpana',
    author_email="kalpana8642@gmail.com",
    packages=find_packages(),  #it will fine the folders which has "__init__.py" within them and treat them as
    install_requires=get_requirements('requirements.txt')  #we have defined this function above
)



