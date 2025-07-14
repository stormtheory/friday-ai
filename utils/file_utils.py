def extract_text_from_json(data, indent=0):
    output = ""
    if isinstance(data, dict):
        for key, value in data.items():
            output += f"{'  '*indent}{key}: "
            output += extract_text_from_json(value, indent+1)
    elif isinstance(data, list):
        for item in data:
            output += extract_text_from_json(item, indent+1)
    else:
        output += f"{str(data)}\n"
    return output
