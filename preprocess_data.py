import argparse
import os
import os.path


def process_data(source_dir, target_dir, target_separator):
    # Iterate over files in source_dir and rename equivalent file in target_dir
    # If strings of files are the same
    target_files = {_process_filename(f, '.mp3', target_separator): os.path.join(target_dir, f)
                    for f in os.listdir(target_dir)
                    if os.path.isfile(os.path.join(target_dir, f)) and not f.startswith('.')}
    (_, _, source_files) = next(os.walk(source_dir))
    for source_f in source_files:
        processed_fname = _process_filename(source_f, '.lab')
        if processed_fname in target_files:
            old_name = target_files[processed_fname]
            new_name = os.path.join(os.path.dirname(old_name), source_f.replace('.lab', '.mp3'))
            print(f"Renamed '{old_name}' to '{new_name}'")
            os.rename(old_name, new_name)


def _process_filename(filename, ext, separator=' '):
    f = filename.replace(ext, '')
    _, result = f.split(separator, maxsplit=1)
    return result.lstrip().lower()


def _check_args(args):
    if not os.path.isdir(args.source) and not os.path.isdir(args.target):
        raise ValueError("source or/and target is not a directory")
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
    args = parser.parse_args()
    _check_args(args)
    process_data(source_dir=args.source,
                 target_dir=args.target,
                 target_separator=args.target_separator)


if __name__ == '__main__':
    main()
