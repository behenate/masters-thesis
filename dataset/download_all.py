from downloaders import (
    spam_ham,
    phishing_email,
    fraud_email,
    fraudulent_email_corpus,
    spam_assassin,
    trec_2007,
    ceas_2008,
)

DOWNLOADERS = [
    ("spam_ham", spam_ham),
    ("phishing_email", phishing_email),
    ("fraud_email", fraud_email),
    ("fraudulent_email_corpus", fraudulent_email_corpus),
    ("spam_assassin", spam_assassin),
    ("trec_2007", trec_2007),
    ("ceas_2008", ceas_2008),
]


def download_all() -> dict[str, str | None]:
    print("=== Downloading all datasets ===\n")
    results = {}

    for name, downloader in DOWNLOADERS:
        print(f"--- {name} ---")
        results[name] = downloader.download()
        print()

    print("=== Summary ===")
    for name, path in results.items():
        print(f"  {name}: {path or 'FAILED'}")

    return results


if __name__ == "__main__":
    download_all()
