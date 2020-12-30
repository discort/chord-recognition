import argparse
import os
import os.path


def process_data(source_dir, target_dir, target_separator, target_format):
    # Iterate over files in source_dir and rename equivalent file in target_dir
    # If strings of files are the same
    target_files = {}
    for root, dires, files in os.walk(target_dir):
        for f in files:
            if f.endswith(target_format):
                target_files[_process_filename(f, target_format, '.')] = os.path.join(root, f)

    #import pudb; pudb.set_trace()
    for root, dires, source_files in os.walk(source_dir):
        for source_f in source_files:
            if not source_f.endswith('.lab'):
                continue
            processed_fname = _process_filename(source_f, '.lab', '-')
            if processed_fname in target_files:
                old_name = target_files[processed_fname]
                new_name = os.path.join(os.path.dirname(old_name), source_f.replace('.lab', target_format))
                print(f"Renamed '{old_name}' to '{new_name}'")
                os.rename(old_name, new_name)


def _process_filename(filename, ext, separator=' '):
    f = filename.replace(ext, '')
    if separator:
        _, f = f.split(separator, maxsplit=1)
    if ext == '.lab':
        f = f.replace('_', ' ')
    return f.lstrip('_').strip().lower()


def _check_args(args):
    if not os.path.isdir(args.source) and not os.path.isdir(args.target):
        raise ValueError("source or/and target is not a directory")

    if args.target_format not in ('.mp3', '.flac'):
        raise ValueError("File format must be either.flac or .mp3")
    # ToDo:
    # - Check if there is same amout of files in source and target


def main():
    """
    Usage:
        > python preprocess_data.py data/carole_king/chordlabs data/carole_king/mp3
        > Renamed 'data/carole_king/mp3/01. I Feel the Earth Move.mp3'
        > to 'data/carole_king/mp3/01 I Feel The Earth Move.mp3'
    """
    parser = argparse.ArgumentParser(
        description='Rename target files by source if there is match between their strings')
    parser.add_argument("source", help="Specify directory with source files")
    parser.add_argument("target", help="Specify directory with target files")
    parser.add_argument("--target-separator", help="Optional separator in target files")
    parser.add_argument("--target-format", help="File format for target. mp3 by default",
                        default='.mp3')
    args = parser.parse_args()
    _check_args(args)
    process_data(source_dir=args.source,
                 target_dir=args.target,
                 target_separator=args.target_separator,
                 target_format=args.target_format)


if __name__ == '__main__':
    main()
