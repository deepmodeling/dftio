import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dftio import __version__
from .io.parse import ParserRegister
from tqdm import tqdm
from .logger import set_log_handles

def get_ll(log_level: str) -> int:
    """Convert string to python logging level.

    Parameters
    ----------
    log_level : str
        allowed input values are: DEBUG, INFO, WARNING, ERROR, 3, 2, 1, 0

    Returns
    -------
    int
        one of python logging module log levels - 10, 20, 30 or 40
    """
    if log_level.isdigit():
        int_level = (4 - int(log_level)) * 10
    else:
        int_level = getattr(logging, log_level)

    return int_level

def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="dftio is to assist machine learning communities to transcript DFT output into a format that is easy to read or used by machine learning models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-v', '--version', 
                        action='version', version=f'%(prog)s {__version__}', help="show the dftio's version number and exit")


    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # log parser
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser_log.add_argument(
        "-ll",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO "
             "and 3=DEBUG",
    )

    parser_log.add_argument(
        "-lp",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
             "only be output to console",
    )

    # config parser
    parser_parse = subparsers.add_parser(
        "parse",
        parents=[parser_log],
        help="parse dataset from DFT output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_parse.add_argument(
        "-m",
        "--mode",
        type=str,
        default="abacus",
        help="The name of the DFT software.",
    )

    parser_parse.add_argument(
        "-r",
        "--root",
        type=str,
        default="./",
        help="The root directory of the DFT files.",
    )

    parser_parse.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="frame",
        help="The prefix of the DFT files under root.",
    )

    parser_parse.add_argument(
        "-o",
        "--outroot",
        type=str,
        default="./",
        help="The output root directory.",
    )

    parser_parse.add_argument(
        "-f",
        "--format",
        type=str,
        default="dat",
        help="The output root directory.",
    )

    parser_parse.add_argument(
        "-ham",
        "--hamiltonian",
        action="store_true",
        help="Whether to parse the Hamiltonian matrix.",
    )

    parser_parse.add_argument(
        "-ovp",
        "--overlap",
        action="store_true",
        help="Whether to parse the Overlap matrix",
    )
    parser_parse.add_argument(
        "-dm",
        "--density_matrix",
        action="store_true",
        help="Whether to parse the Density matrix",
    )

    parser_parse.add_argument(
        "-eig",
        "--eigenvalue",
        action="store_true",
        help="Whether to parse the kpoints and eigenvalues",
    )

    return parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments and convert argument strings to objects.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv

    Returns
    -------
    argparse.Namespace
        the populated namespace
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)

    return parsed_args

def main():
    args = parse_args()

    if args.command not in (None, "train", "test", "run"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == "parse":
        parser = ParserRegister(
            **dict_args
        )

        for i in tqdm(range(len(parser)), desc="Parsing the DFT files: "):
            parser.write(idx=i, **dict_args)
        


if __name__ == "__main__":
    main()