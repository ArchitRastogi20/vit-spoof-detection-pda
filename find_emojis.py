import os
import sys
import re

#  Rough-but-good emoji regex (covers most emoji in modern use)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess, symbols
    "\U0001FA70-\U0001FAFF"  # more symbols
    "\u2600-\u26FF"          # misc symbols
    "\u2700-\u27BF"          # dingbats
    "]+"
)

# Extensions you care about (edit as needed)
DEFAULT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".json", ".yml", ".yaml",
    ".md", ".txt",
}


def has_valid_ext(filename, allowed_exts):
    _, ext = os.path.splitext(filename)
    return ext.lower() in allowed_exts


def scan_file(path):
    results = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                match = EMOJI_PATTERN.search(line)
                if match:
                    results.append((lineno, line.rstrip("\n")))
    except (UnicodeDecodeError, PermissionError, OSError):
        pass
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_emojis.py <directory> [ext1 ext2 ...]")
        sys.exit(1)

    root = sys.argv[1]

    if len(sys.argv) > 2:
        allowed_exts = {("." + e.lstrip(".")) for e in sys.argv[2:]}
    else:
        allowed_exts = DEFAULT_EXTS

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not has_valid_ext(filename, allowed_exts):
                continue

            full_path = os.path.join(dirpath, filename)
            hits = scan_file(full_path)
            for lineno, line in hits:
                print(f"{full_path}:{lineno}: {line}")


if __name__ == "__main__":
    main()
