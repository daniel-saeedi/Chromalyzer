
import os
import shutil
from setuptools import setup, find_packages, Command

with open("app/README.md", "r") as f:
    long_description = f.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for dirpath in ('./build', './dist', './*.egg-info'):
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
        print("Cleaned up the project directories.")

setup(
    name='chromalyzer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'loguru',
        'tqdm'
    ],
    python_requires=">=3.10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmdclass={
        'clean': CleanCommand,
    },

    entry_points={
        'console_scripts': [
            'chromalyzer=app.chromalyzer.__main__:main',
        ],
    },
)