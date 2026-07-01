from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".onnx", ".engine", ".trt", ".weights", ".ckpt",
    ".pkl", ".pickle", ".npy", ".npz",
    ".mp4", ".avi", ".mov", ".mkv",
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".zip", ".rar", ".7z", ".tar", ".gz",
}

FORBIDDEN_NAME_ENDINGS = {
    ".pth.tar",
}

FORBIDDEN_DIRS = {
    "__pycache__", ".ipynb_checkpoints", ".git",
    "data", "datasets", "frames", "videos", "references",
    "outputs", "results", "logs", "cache", "debug", "visualizations",
    "models", "downloaded_models",
}

def is_forbidden(path: Path) -> bool:
    if any(part in FORBIDDEN_DIRS for part in path.parts):
        return True

    lower_name = path.name.lower()

    if any(lower_name.endswith(ending) for ending in FORBIDDEN_NAME_ENDINGS):
        return True

    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True

    return False

def main():
    violations = []

    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)

        if path.is_file() and is_forbidden(rel):
            violations.append(str(rel))

    if violations:
        print("Forbidden files found:")
        for item in violations:
            print(" -", item)
        sys.exit(1)

    print("Repository validation passed. No forbidden files found.")

if __name__ == "__main__":
    main()
