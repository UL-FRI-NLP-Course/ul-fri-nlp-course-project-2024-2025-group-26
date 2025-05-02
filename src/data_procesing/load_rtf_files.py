import os
import glob
import logging
import re
import subprocess
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

def decode_rtf_escapes(rtf_content: str) -> str:
    """
    Decodes RTF hex escapes (\'xx) assuming Windows-1250 encoding.
    """
    def decode_match(match):
        hex_value = match.group(1)
        byte = bytes.fromhex(hex_value)
        return byte.decode('windows-1250')

    # Replace all \'xx sequences
    decoded_content = re.sub(r"\\'([0-9a-fA-F]{2})", decode_match, rtf_content)

    return decoded_content

def convert_fixed_rtf_to_markdown(rtf_text: str) -> str:
    """
    Converts decoded RTF string to Markdown using Pandoc CLI.
    """
    result = subprocess.run(
        ["pandoc", "--from", "rtf", "--to", "markdown"],
        input=rtf_text.encode('utf-8'),
        capture_output=True,
        text=False  
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pandoc failed: {result.stderr.decode('utf-8', errors='replace')}")

    return result.stdout.decode('utf-8', errors='replace')

def save_rtf_files_as_json(promet_folder: str, promet_subfolders: list[str]) -> None:
    file_index = 0
    all_rtf_files = []

    promet_subfolders.sort()

    for subfolder in promet_subfolders:
        folder_path = os.path.join(promet_folder, subfolder)
        if not os.path.exists(folder_path):
            logger.warning(f"Folder {folder_path} does not exist.")
            continue

        rtf_files = glob.glob(os.path.join(folder_path, "**", "*.rtf"), recursive=True)
        for rtf_file in rtf_files:
            full_path = os.path.abspath(rtf_file)
            if os.path.basename(full_path).startswith("~$"):
                continue
            if not os.path.isfile(full_path):
                logger.warning(f"File {full_path} does not exist.")
                continue
            all_rtf_files.append((file_index, rtf_file, full_path))
            file_index += 1

    all_rtf_files.sort(key=lambda x: x[0])
    records = []

    # Regex that matches dates in format like "30. 04. 2022 18.30"
    date_pattern = re.compile(
    r"(?P<day>\d{1,2})\.\s*(?P<month>\d{1,2})\.\s*(?P<year>\d{4})\s*(?P<hour>\d{1,2})\.(?P<minute>\d{2})"
)

    time_sum = 0
    for file_index, rtf_file, full_path in all_rtf_files:
        t1 = time.time()
        logger.info(f"Converting file {file_index}: {full_path}")

        parts = os.path.normpath(full_path).split(os.sep)
        path2 = os.sep.join(parts[-4:])
        
        try:
            with open(full_path, 'r', encoding='latin1') as file:
                rtf_content = file.read()

            fixed_rtf_content = decode_rtf_escapes(rtf_content)
            markdown_content = convert_fixed_rtf_to_markdown(fixed_rtf_content)

            # Extract date from first line of the markdown file.
            lines = markdown_content.splitlines()
            if not lines:
                logger.error(f"No content in markdown from {full_path}")
                continue

            lines = markdown_content.splitlines()
            date_line = None
            for line in lines:
                stripped_line = line.strip()
                if stripped_line:
                    date_line = stripped_line
                    break
            if not date_line:
                logger.error(f"No non-empty content found in markdown from {full_path}")
                continue

            date_match = date_pattern.search(date_line)
            if not date_match:
                logger.error(f"No valid date found in first line of {full_path}: {date_line}")
                continue

            try:
                day = int(date_match.group("day"))
                month = int(date_match.group("month"))
                year = int(date_match.group("year"))
                hour = int(date_match.group("hour"))
                minute = int(date_match.group("minute"))
                dt = datetime(year, month, day, hour, minute)
            except Exception as ve:
                logger.error(f"Date parsing failed for {full_path}: {ve}")
                continue

            record = {
                "id": file_index,
                "date": dt.isoformat(),
                "markdown": markdown_content,
                "file_path": path2,
            }
            records.append(record)
            t2 = time.time()
            time_sum += (t2 - t1)
            if file_index > 0 and file_index % 10 == 0:
                time_per_file = time_sum / file_index
                remaining_files = len(all_rtf_files) - file_index
                estimated_time = time_per_file * remaining_files
                estimated_time_s = time.strftime("%H:%M:%S", time.gmtime(estimated_time))
                print(f"Processed {file_index}: out of {len(all_rtf_files)} time left {estimated_time_s}",end="\r")


        except Exception as e:
            logger.error(f"Failed to convert {full_path}: {e}")

    # Sort records by date, so that you can perform range queries efficiently.
    records.sort(key=lambda x: x["date"])
    
    for new_index, record in enumerate(records):
        record["id"] = new_index

    # Save the records to a JSON file. (You can later load this file and filter by date.)
    output_file = "rtf_data.json"
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(records, out_file, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(records)} records to {output_file}")

if __name__ == "__main__":
    PROMET_FOLDER = r"C:\Users\rozma\Downloads\RTVSlo\RTVSlo\Podatki"
    PROMET_SUBFOLDERS = [r'Promet 2022', r'Promet 2023', r'Promet 2024']
    save_rtf_files_as_json(PROMET_FOLDER, PROMET_SUBFOLDERS)