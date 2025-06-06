"""
Cleaning original data.json to have only chapterNum, subChapterNum, sectionNum, title, text
"""


import json
import os

def clean_document(json_data):
    cleaned_data = []

    for chapter in json_data:
        if chapter.get('name') != 'Chapter':
            continue

        cleaned_chapter = {
            'chapterNum': chapter.get('chapterNum'),
            'title': chapter.get('title'),
            'subchapters': []
        }

        for subchapter in chapter.get('subchapters', []):
            cleaned_subchapter = {
                'subChapterNum': subchapter.get('subChapterNum'),
                'title': subchapter.get('title'),
                'sections': []
            }

            for section in subchapter.get('sections', []):
                cleaned_section = {
                    'sectionNum': section.get('sectionNum'),
                    'title': section.get('title'),
                    'text': section.get('text')
                }
                cleaned_subchapter['sections'].append(cleaned_section)

            cleaned_chapter['subchapters'].append(cleaned_subchapter)

        cleaned_data.append(cleaned_chapter)

    return cleaned_data


# Example usage
if __name__ == "__main__":
    input_file = 'OREC/statutes.json'
    output_file = 'OREC/cleaned_statutes.json'

    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Input file '{input_file}' not found. Please ensure it exists in the same directory as this script.")

        # Load JSON data
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Extract cleaned data
        cleaned_data = clean_document(data)

        # Save cleaned data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(cleaned_data, file, indent=2)

        print(f"Successfully saved cleaned data to '{output_file}'")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file}'. Details: {e}")
    except PermissionError:
        print(f"Error: Permission denied when writing to '{output_file}'. Check file permissions.")
    except Exception as e:
        print(f"Unexpected error: {e}")