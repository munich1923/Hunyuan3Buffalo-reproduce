"""Download the pinned ABO/HSSD TRELLIS metadata records."""
import json

from core import cfg, metadata, setup


if __name__ == "__main__":
    setup()
    rows = metadata(cfg())
    print(json.dumps({"metadata_rows": len(rows), "assets_downloaded": False}, indent=2))
