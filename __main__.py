"""
ChromClassifier - A python package for classifying 2D Gas Chromatography (GCxGC) data.
Author: Daniel Saeedi
"""

def main():
    import argparse
    import warnings

    import ChromClassifier

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ChromClassifier {ChromClassifier.__version__}",
    )

    # Suppress some warnings
    warnings.filterwarnings("ignore", message="^.* socket cannot be initialized.*$")

    # modules = {
    #     'build': ChromClassifier.build,
    #     'train': ChromClassifier.train,
    #     'predict': ChromClassifier.predict,
    #     'evaluate': ChromClassifier.evaluate,
    #     'visualize': ChromClassifier.visualize,
    # }

