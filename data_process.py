from docx import Document
import json

# Load the document (replace 'file_path.docx' with the path to your document)
doc = Document('B-dataset.docx')

# Initialize an empty list to store structured data
structured_data = []

# Get all paragraphs from the document
paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

# Ensure the number of paragraphs is a multiple of 3 (Text, Intent, Entities)
if len(paragraphs) % 3 == 0:
    for i in range(0, len(paragraphs), 3):
        text = paragraphs[i].split("Text: ")[-1]
        intent = paragraphs[i + 1].split("Intent: ")[-1]
        entities_text = paragraphs[i + 2].split("Entities: ")[-1]
        
        # Evaluate Entities from the text as a Python dictionary
        # entities = entities_text.split()
        
        # Create a structured dictionary for each example and append to structured_data list
        example = {
            "text": text,
            "intent": intent,
            "entities": entities_text
        }
        structured_data.append(example)

    # Save the structured data to a JSON file
    with open('B_data.json', 'w') as json_file:
        json.dump(structured_data, json_file, indent=2)

    print("Conversion completed. Data saved to 'structured_data.json'")
else:
    print("Error: The document does not contain a valid number of paragraphs for processing.")

