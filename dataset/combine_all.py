from combiners import spam_detection, spam_2007_2008

COMBINERS = [
    ("spam_detection", spam_detection),
    ("spam_2007_2008", spam_2007_2008),
]


def combine_all() -> dict[str, str | None]:
    print("=== Combining all datasets ===\n")
    results = {}

    for name, combiner in COMBINERS:
        print(f"--- {name} ---")
        results[name] = combiner.combine()
        print()

    print("=== Summary ===")
    for name, path in results.items():
        print(f"  {name}: {path or 'FAILED'}")

    return results


if __name__ == "__main__":
    combine_all()
