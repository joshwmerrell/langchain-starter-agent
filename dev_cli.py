"""Start the CLI with source-development access and development threads."""

import os

os.environ["WALLACE_DEV_MODE"] = "1"

from main import main  # noqa: E402


if __name__ == "__main__":
    main()
