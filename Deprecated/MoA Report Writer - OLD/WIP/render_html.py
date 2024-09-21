to_insert = ""

for i, highlight in enumerate(highlights):
    to_insert += f"""<h1>Highlighted Response {i+1}</h1>
    <p>
        {highlight}
    </p>\n
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