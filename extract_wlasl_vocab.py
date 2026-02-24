import json
import os

def main():
    json_path = 'WLASL_v0.3.json'
    output_path = 'actions.txt'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Reading {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        glosses = set()
        for entry in data:
            if 'gloss' in entry:
                glosses.add(entry['gloss'])
        
        sorted_glosses = sorted(list(glosses))
        count = len(sorted_glosses)
        
        print(f"Found {count} unique glosses.")
        
        with open(output_path, 'w') as f:
            for gloss in sorted_glosses:
                f.write(gloss + '\n')
                
        print(f"Successfully wrote {count} glosses to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
