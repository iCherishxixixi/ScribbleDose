import os
import argparse
import tempfile
import numpy as np
from tqdm import tqdm


def load_npz_dict(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    if "arr_0" not in data:
        raise KeyError(f"No 'arr_0' found in {npz_path}. Existing keys: {list(data.keys())}")

    obj = data["arr_0"]

    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()

    return obj.item()


def save_npz_dict(save_path, data_dict):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(save_path))
    os.close(tmp_fd)

    np.savez_compressed(tmp_path, arr_0=data_dict)
    os.replace(tmp_path, save_path)


def is_scribble_or_supervoxel_key(key):
    return key.endswith("_scribble") or key == "supervoxels"


def collect_npz_files(root):
    npz_files = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".npz"):
                path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(path, root)
                npz_files.append((path, rel_path))

    return sorted(npz_files, key=lambda x: x[1])


def compare_values(a, b):
    """
    Compare two values from npz dict.
    Supports numpy arrays, scalars, strings, and simple objects.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b)
        except Exception:
            return False

    return a == b


def validate_clean_dict(clean_dict, rel_path):
    bad_keys = [
        k for k in clean_dict.keys()
        if is_scribble_or_supervoxel_key(k)
    ]

    if bad_keys:
        raise ValueError(
            f"{rel_path} in clean AAPM_annotation still contains scribble/supervoxel keys: "
            f"{bad_keys[:20]}"
        )


def validate_scribble_dict(scribble_dict, rel_path):
    bad_keys = [
        k for k in scribble_dict.keys()
        if not is_scribble_or_supervoxel_key(k)
    ]

    if bad_keys:
        raise ValueError(
            f"{rel_path} in ScribbleDose_annotation contains unexpected keys: "
            f"{bad_keys[:20]}"
        )


def merge_one_case(
    clean_npz_path,
    scribble_npz_path,
    output_npz_path,
    rel_path,
    apply=False,
    overwrite_keys=False,
):
    clean_dict = load_npz_dict(clean_npz_path)
    scribble_dict = load_npz_dict(scribble_npz_path)

    validate_clean_dict(clean_dict, rel_path)
    validate_scribble_dict(scribble_dict, rel_path)

    merged_dict = clean_dict.copy()

    collision_keys = []

    for key, value in scribble_dict.items():
        if key in merged_dict:
            collision_keys.append(key)

            if not overwrite_keys:
                raise KeyError(
                    f"Key collision in {rel_path}: '{key}' already exists in clean npz. "
                    f"Use --overwrite_keys if you want to overwrite it."
                )

        merged_dict[key] = value

    n_scribble = sum(k.endswith("_scribble") for k in scribble_dict.keys())
    has_supervoxels = "supervoxels" in scribble_dict

    expected_num_keys = len(clean_dict) + len(scribble_dict) - len(collision_keys)

    if len(merged_dict) != expected_num_keys:
        raise RuntimeError(
            f"Unexpected merged key number in {rel_path}: "
            f"clean={len(clean_dict)}, scribble={len(scribble_dict)}, "
            f"merged={len(merged_dict)}, expected={expected_num_keys}"
        )

    if apply:
        save_npz_dict(output_npz_path, merged_dict)

    return {
        "n_clean_keys": len(clean_dict),
        "n_scribble_package_keys": len(scribble_dict),
        "n_merged_keys": len(merged_dict),
        "n_scribble": n_scribble,
        "has_supervoxels": has_supervoxels,
        "collision_keys": collision_keys,
    }


def test_merged_file(output_npz_path, clean_npz_path, scribble_npz_path, rel_path):
    """
    Test whether the merged file contains all clean keys and all scribble/supervoxel keys.
    """
    merged_dict = load_npz_dict(output_npz_path)
    clean_dict = load_npz_dict(clean_npz_path)
    scribble_dict = load_npz_dict(scribble_npz_path)

    missing_clean_keys = []
    wrong_clean_values = []

    for key, value in clean_dict.items():
        if key not in merged_dict:
            missing_clean_keys.append(key)
        elif not compare_values(merged_dict[key], value):
            wrong_clean_values.append(key)

    missing_scribble_keys = []
    wrong_scribble_values = []

    for key, value in scribble_dict.items():
        if key not in merged_dict:
            missing_scribble_keys.append(key)
        elif not compare_values(merged_dict[key], value):
            wrong_scribble_values.append(key)

    if (
        missing_clean_keys
        or wrong_clean_values
        or missing_scribble_keys
        or wrong_scribble_values
    ):
        raise AssertionError(
            f"Merge test failed for {rel_path}\n"
            f"Missing clean keys: {missing_clean_keys[:20]}\n"
            f"Wrong clean values: {wrong_clean_values[:20]}\n"
            f"Missing scribble keys: {missing_scribble_keys[:20]}\n"
            f"Wrong scribble values: {wrong_scribble_values[:20]}"
        )

    return True


def test_against_backup(output_npz_path, backup_npz_path, rel_path):
    """
    Optional strict test.
    If you have the original backed-up AAPM_annotation before splitting,
    this test checks whether the merged npz is exactly equivalent to the backup.
    """
    merged_dict = load_npz_dict(output_npz_path)
    backup_dict = load_npz_dict(backup_npz_path)

    merged_keys = set(merged_dict.keys())
    backup_keys = set(backup_dict.keys())

    missing_keys = sorted(list(backup_keys - merged_keys))
    extra_keys = sorted(list(merged_keys - backup_keys))

    wrong_values = []

    for key in sorted(list(backup_keys & merged_keys)):
        if not compare_values(merged_dict[key], backup_dict[key]):
            wrong_values.append(key)

    if missing_keys or extra_keys or wrong_values:
        raise AssertionError(
            f"Backup comparison failed for {rel_path}\n"
            f"Missing keys compared with backup: {missing_keys[:20]}\n"
            f"Extra keys compared with backup: {extra_keys[:20]}\n"
            f"Wrong values compared with backup: {wrong_values[:20]}"
        )

    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge ScribbleDose annotations back into clean AAPM_annotation npz files. "
            "The script merges *_scribble and supervoxels by matching identical relative filenames."
        )
    )

    parser.add_argument(
        "--clean_root",
        type=str,
        default="AAPM_annotation",
        help="Root directory of clean AAPM_annotation after removing scribbles and supervoxels."
    )

    parser.add_argument(
        "--scribble_root",
        type=str,
        default="ScribbleDose_annotation",
        help="Root directory of ScribbleDose_annotation containing only *_scribble and supervoxels."
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="AAPM_annotation_merged",
        help="Output root directory for merged npz files."
    )

    parser.add_argument(
        "--backup_root",
        type=str,
        default=None,
        help=(
            "Optional root directory of the original backed-up AAPM_annotation before splitting. "
            "If provided, the script will compare merged files with backup files."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write merged files. Without this flag, the script only performs a dry run."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="After merging, test whether merged files contain all clean and scribble keys."
    )

    parser.add_argument(
        "--overwrite_keys",
        action="store_true",
        help="Overwrite keys if collision happens during merging."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each file."
    )

    args = parser.parse_args()

    clean_files = collect_npz_files(args.clean_root)

    print(f"Clean root:    {args.clean_root}")
    print(f"Scribble root: {args.scribble_root}")
    print(f"Output root:   {args.output_root}")
    print(f"Backup root:   {args.backup_root}")
    print(f"Found clean npz files: {len(clean_files)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Test after merge: {args.test}")

    total_processed = 0
    total_missing_scribble_files = 0
    total_error_files = 0
    total_scribble_keys = 0
    total_supervoxels = 0
    total_backup_test_passed = 0
    total_basic_test_passed = 0

    missing_scribble_files = []
    error_files = []

    min_scribble = None
    max_scribble = None

    for clean_npz_path, rel_path in tqdm(clean_files, desc="Merging npz files"):
        scribble_npz_path = os.path.join(args.scribble_root, rel_path)
        output_npz_path = os.path.join(args.output_root, rel_path)

        if not os.path.exists(scribble_npz_path):
            total_missing_scribble_files += 1
            missing_scribble_files.append(rel_path)

            if args.verbose:
                print(f"[Missing] {rel_path}: no corresponding scribble npz found")

            continue

        try:
            info = merge_one_case(
                clean_npz_path=clean_npz_path,
                scribble_npz_path=scribble_npz_path,
                output_npz_path=output_npz_path,
                rel_path=rel_path,
                apply=args.apply,
                overwrite_keys=args.overwrite_keys,
            )

            total_processed += 1
            total_scribble_keys += info["n_scribble"]
            total_supervoxels += int(info["has_supervoxels"])

            if min_scribble is None or info["n_scribble"] < min_scribble:
                min_scribble = info["n_scribble"]

            if max_scribble is None or info["n_scribble"] > max_scribble:
                max_scribble = info["n_scribble"]

            if args.verbose:
                print(
                    f"[OK] {rel_path}: "
                    f"clean_keys={info['n_clean_keys']}, "
                    f"scribble_package_keys={info['n_scribble_package_keys']}, "
                    f"merged_keys={info['n_merged_keys']}, "
                    f"scribble_keys={info['n_scribble']}, "
                    f"supervoxels={info['has_supervoxels']}"
                )

            if args.apply and args.test:
                test_merged_file(
                    output_npz_path=output_npz_path,
                    clean_npz_path=clean_npz_path,
                    scribble_npz_path=scribble_npz_path,
                    rel_path=rel_path,
                )
                total_basic_test_passed += 1

                if args.backup_root is not None:
                    backup_npz_path = os.path.join(args.backup_root, rel_path)

                    if not os.path.exists(backup_npz_path):
                        raise FileNotFoundError(f"Backup file not found: {backup_npz_path}")

                    test_against_backup(
                        output_npz_path=output_npz_path,
                        backup_npz_path=backup_npz_path,
                        rel_path=rel_path,
                    )
                    total_backup_test_passed += 1

        except Exception as e:
            total_error_files += 1
            error_files.append((rel_path, str(e)))

            if args.verbose:
                print(f"[Error] {rel_path}: {e}")

    avg_scribble = total_scribble_keys / total_processed if total_processed > 0 else 0

    print("\n========== Summary ==========")
    print(f"Mode:                         {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Total clean npz files found:   {len(clean_files)}")
    print(f"Processed files:               {total_processed}")
    print(f"Missing scribble files:        {total_missing_scribble_files}")
    print(f"Error files:                   {total_error_files}")
    print(f"Total scribble keys merged:    {total_scribble_keys}")
    print(f"Average scribble keys/file:    {avg_scribble:.2f}")
    print(f"Min scribble keys/file:        {min_scribble}")
    print(f"Max scribble keys/file:        {max_scribble}")
    print(f"Files with supervoxels:        {total_supervoxels}")

    if args.apply and args.test:
        print(f"Basic merge tests passed:      {total_basic_test_passed}")

        if args.backup_root is not None:
            print(f"Backup comparison passed:      {total_backup_test_passed}")

    if len(missing_scribble_files) > 0:
        print("\nExamples of missing scribble files:")
        for name in missing_scribble_files[:20]:
            print(f"  - {name}")

    if len(error_files) > 0:
        print("\nError files:")
        for name, err in error_files[:20]:
            print(f"  - {name}: {err}")

    if not args.apply:
        print("\nThis was a dry run. No files were written.")
        print("To actually merge files, rerun with --apply.")
    else:
        print("\nMerging finished.")
        print("Merged files were saved to:")
        print(f"  {args.output_root}")


if __name__ == "__main__":
    main()