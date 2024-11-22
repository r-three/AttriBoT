from setuptools import setup, find_packages


# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='context-attribution-simple',
    version='1.0.0',
    packages=find_packages(),

    # Include additional files like non-code resources or configuration files
    include_package_data=True,

    # Provide a short description of your package
    description='',

    # Add your author information
    author='',
    author_email='',

    # Add project URL if any
    url='',

    # Add any required dependencies
    install_requires=requirements,
    entry_points={
    },

    # Add classifiers to specify the audience and maturity of your package
    classifiers=[
    ],
)

