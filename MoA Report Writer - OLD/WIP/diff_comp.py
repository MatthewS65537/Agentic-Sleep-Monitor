import difflib
import re

def highlight_similarities(text1, text2, id=1):
    words1 = text1.split()
    words2 = text2.split()
    
    matcher = difflib.SequenceMatcher(None, words1, words2)
    highlighted1 = []
    highlighted2 = []
    
    highlight_start, highlight_end = f"<span style=\"highlight{id}\">", f"</span>"
    
    for opcode in matcher.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag == 'equal':
            highlighted1.append(f"{highlight_start}{' '.join(words1[i1:i2])}{highlight_end}")
            highlighted2.append(f"{highlight_start}{' '.join(words2[j1:j2])}{highlight_end}")
        else:
            highlighted1.extend(words1[i1:i2])
            highlighted2.extend(words2[j1:j2])
    
    return ' '.join(highlighted1), ' '.join(highlighted2)

all_highlights = []

# Iterate through all responses
for i in range(1, 3):
    with open(f"response_{i}.md", "r") as f:
        cur_response = f.read()

    with open("final_response.md", "r") as f:
        final_response = f.read()

    highlighted_response, highlighted_final = highlight_similarities(cur_response, final_response, id=i)

    all_highlights.append([highlighted_response, highlighted_final])

to_insert = ""

for i, highlight in enumerate(all_highlights):
    to_insert += f"""<h1>Highlighted Response {i+1}</h1>
    <p>
        {highlight[0]}
    </p>
    <h1>Highlighted Final Response</h1>
    <p>
        {highlight[1]}
    </p>
    """

code="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baseline HTML</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .highlight1 { background-color: #FFB3BA; }
        .highlight2 { background-color: #BAFFC9; }
        .highlight3 { background-color: #BAE1FF; }
        .highlight4 { background-color: #FFFFBA; }
    </style>
</head>
<body>""" + to_insert + """</body>
</html>
"""

with open("main.html", "w") as f:
    f.write(code)