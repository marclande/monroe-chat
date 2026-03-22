"""
Download Monroe Institute transcripts from archive.org
======================================================
Downloads text transcripts from all available sections beyond the Explorers Project.
"""

import os
import sys
import time
import urllib.request
import urllib.error
import json

sys.stdout.reconfigure(encoding="utf-8")

# Base download directory
BASE_DIR = "C:/Users/marc.lande/Downloads/monroe_archive/monroe_archive/transcripts_expanded"

# Archive.org items that have transcripts
ARCHIVE_ITEMS = [
    # Robert A. Monroe sections
    {"id": "ram-interviews", "category": "RAM_Interviews"},
    {"id": "ram-talks", "category": "RAM_Talks"},
    {"id": "ram-gateway-voyage-talks-day-1-saturday", "category": "RAM_Gateway_Voyage"},
    {"id": "ram-gateway-voyage-talks-day-2-sunday", "category": "RAM_Gateway_Voyage"},
    {"id": "ram-gateway-voyage-talks-day-3-monday", "category": "RAM_Gateway_Voyage"},
    {"id": "ram-gateway-voyage-talks-day-4-tuesday", "category": "RAM_Gateway_Voyage"},
    {"id": "ram-gateway-voyage-talks-day-5-wednesday", "category": "RAM_Gateway_Voyage"},
    {"id": "ram-gateway-talks-day-6-thursday", "category": "RAM_Gateway_Voyage"},
    {"id": "ram-gateway-voyage-talks-other", "category": "RAM_Gateway_Voyage"},
    {"id": "ram-gateway-voyage-inscom-1983-12", "category": "RAM_INSCOM"},
    {"id": "ram-gateway-voyage-inscom-1984-01", "category": "RAM_INSCOM"},
    {"id": "ram-graduate-gateway-talks-june-1984", "category": "RAM_Graduate_Gateway"},
    {"id": "ram-guidelines-talks-day-1-saturday", "category": "RAM_Guidelines"},
    {"id": "ram-guidelines-talks-day-2-sunday", "category": "RAM_Guidelines"},
    {"id": "ram-guidelines-talks-day-6-thursday", "category": "RAM_Guidelines"},
]


def get_file_list(item_id):
    """Get list of files for an archive.org item via metadata API."""
    url = f"https://archive.org/metadata/{item_id}/files"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("result", [])
    except Exception as e:
        print(f"  Error fetching file list for {item_id}: {e}")
        return []


def sanitize_filename(name):
    """Remove characters invalid in Windows filenames."""
    for ch in ['*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '')
    return name.strip()


def download_file(item_id, filename, save_path):
    """Download a single file from archive.org."""
    url = f"https://archive.org/download/{item_id}/{urllib.request.quote(filename)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(save_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"    Error downloading {filename}: {e}")
        return False


def main():
    print("=" * 70)
    print("Monroe Institute Expanded Archives — Transcript Downloader")
    print("=" * 70)

    os.makedirs(BASE_DIR, exist_ok=True)

    total_downloaded = 0
    total_skipped = 0

    for item in ARCHIVE_ITEMS:
        item_id = item["id"]
        category = item["category"]
        cat_dir = os.path.join(BASE_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)

        print(f"\n--- {item_id} -> {category} ---")
        files = get_file_list(item_id)

        if not files:
            print("  No files found or error.")
            continue

        # Look for transcript files: PDFs with "Transcript" in name, or _djvu.txt files
        transcript_files = []
        for f in files:
            name = f.get("name", "")
            # Prefer PDF transcripts, also grab djvu.txt as backup
            if "(Transcript)" in name and name.endswith(".pdf"):
                transcript_files.append(name)
            elif "(Outline)" in name and name.endswith(".pdf"):
                transcript_files.append(name)
            elif name.endswith("_djvu.txt"):
                transcript_files.append(name)

        if not transcript_files:
            print(f"  No transcript files found among {len(files)} files")
            continue

        print(f"  Found {len(transcript_files)} transcript/outline files")

        for filename in transcript_files:
            safe_name = sanitize_filename(filename)
            save_path = os.path.join(cat_dir, safe_name)
            if os.path.exists(save_path):
                total_skipped += 1
                continue

            print(f"    Downloading: {filename}")
            if download_file(item_id, filename, save_path):
                total_downloaded += 1
                time.sleep(0.5)  # Be nice to archive.org
            else:
                print(f"    FAILED: {filename}")

    print(f"\n{'=' * 70}")
    print(f"DONE: {total_downloaded} files downloaded, {total_skipped} skipped (already exist)")
    print(f"Files saved to: {BASE_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
