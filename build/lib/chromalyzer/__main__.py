import argparse
import warnings

# import chromalyzer


# def main():

#     parser = argparse.ArgumentParser(
#         description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
#     )
#     parser.add_argument(
#         "--version",
#         action="version",
#         version=f"Chromalyzer {chromalyzer.__version__}",
#     )

#     # Suppress some warnings
#     warnings.filterwarnings("ignore", message="^.* socket cannot be initialized.*$")

    

#     modules = {
#         "extract_heatmaps": chromalyzer.extract_heatmaps,
#     }

#     subparsers = parser.add_subparsers(title="Choose a module",)
#     subparsers.required = "True"

#     for key in modules:
#         module_parser = subparsers.add_parser(
#             key,
#             description=modules[key].__doc__,
#             formatter_class=argparse.RawTextHelpFormatter,
#         )
#         modules[key].add_args(module_parser)
#         module_parser.set_defaults(func=modules[key].main)

#     try:
#         args = parser.parse_args()
#         args.func(args)
#     except TypeError:
#         parser.print_help()

def main():
    print("Chromalyzer is running")


if __name__ == "__main__":
    main()