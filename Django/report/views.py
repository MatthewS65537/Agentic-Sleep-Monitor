from django.shortcuts import render
from markdown import markdown
import os

def display_report(request):
    # Read the markdown file
    md_file_path = os.path.join(os.path.dirname(__file__), 'static', 'md', 'report.md')
    with open(md_file_path, 'r') as file:
        md_content = file.read()
    
    # Convert markdown to HTML
    html_content = markdown(md_content, extensions=['extra'])
    
    context = {
        'report_content': html_content
    }
    return render(request, 'report/report.html', context)