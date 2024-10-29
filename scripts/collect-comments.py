import re
import os
import json
import sys

# Regular expressions for capturing struct definitions and their fields with comments
struct_regex = re.compile(r"pub struct (\w+) \{(.*?)^\s*\}", re.DOTALL | re.MULTILINE)
field_regex = re.compile(r"((?:\s*///\s*.*?\n)+)\s*(pub\s+)?(\w+):\s+([\w:<>]+),", re.DOTALL)

def extract_structs_from_rust_file(file_content):
    structs = {}
    
    # Iterate over each struct in the file
    for struct_match in struct_regex.finditer(file_content):
        struct_name = struct_match.group(1)
        struct_body = struct_match.group(2)
        
        fields = {}
        
        # Iterate over each field in the struct body
        for field_match in field_regex.finditer(struct_body):
            raw_comment = field_match.group(1).strip()
            field_name = field_match.group(3).strip()
            field_type = field_match.group(4).strip()
            
            # Join multiple lines of `///` comments with newline
            comment = "\n".join([line.strip()[3:].strip() for line in raw_comment.splitlines()])
            
            fields[field_name] = {"#": comment, "type": field_type}
        
        structs[struct_name] = fields
    return structs

def resolve_struct_recursive(struct_name, structs_metadata):
    if struct_name not in structs_metadata:
        return {}

    resolved_fields = {}
    fields = structs_metadata[struct_name]

    for field_name, field_metadata in fields.items():
        comment = field_metadata["#"]
        field_type = field_metadata["type"]

        # Check if field type matches another struct, if so, recurse
        if field_type in structs_metadata:
            resolved_fields[field_name] = {
                "#": comment,
                **resolve_struct_recursive(field_type, structs_metadata)
            }
        else:
            resolved_fields[field_name] = {"#": comment}

    return resolved_fields

def process_rust_files(file_list):
    structs_metadata = {}
    for file_path in file_list:
        if os.path.exists(file_path) and file_path.endswith(".rs"):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                structs_in_file = extract_structs_from_rust_file(file_content)
                structs_metadata.update(structs_in_file)
        else:
            print(f"Warning: {file_path} not found or not a Rust file.")
    return structs_metadata

def main():
    # Get the list of Rust files from command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <root_struct_name> <file1.rs> <file2.rs> ...")
        sys.exit(1)

    root_struct_name = sys.argv[1]
    rust_files = sys.argv[2:]
    
    # Process the Rust files and extract struct metadata
    metadata = process_rust_files(rust_files)

    if root_struct_name not in metadata:
        print(f"Error: Struct '{root_struct_name}' not found in the provided files.")
        sys.exit(1)

    # Start the recursive resolution from the root struct
    resolved_metadata = resolve_struct_recursive(root_struct_name, metadata)
    resolved_metadata = {
        "##info##": "Use scripts/regen.sh to re-generate this file",
        **resolved_metadata
    }
    
    # Output the metadata as JSON
    output_file = "llgtrt/src/config_info.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(resolved_metadata, f, indent=1)

    print(f"Metadata written to {output_file}")

if __name__ == "__main__":
    main()