#!/usr/bin/env python3
"""
Documentation PDF Generator
Converts markdown documentation to PDF format
"""

import os
import sys
from pathlib import Path

def install_requirements():
    """Install required packages for PDF generation"""
    try:
        import markdown
        import pdfkit
        import weasyprint
    except ImportError:
        print("Installing required packages...")
        os.system("pip install markdown pdfkit weasyprint")
        print("Packages installed successfully!")

def markdown_to_html(md_file):
    """Convert markdown to HTML"""
    import markdown
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Configure markdown with extensions
    md = markdown.Markdown(extensions=[
        'markdown.extensions.tables',
        'markdown.extensions.fenced_code',
        'markdown.extensions.toc',
        'markdown.extensions.codehilite'
    ])
    
    html_content = md.convert(md_content)
    
    # Add CSS styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Enhanced Nifty Option Prediction System Documentation</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
            }}
            .emoji {{
                font-size: 1.2em;
            }}
            @media print {{
                body {{
                    font-size: 12px;
                }}
                h1 {{
                    page-break-before: always;
                }}
                h1:first-child {{
                    page-break-before: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    return html_template

def html_to_pdf(html_content, output_file):
    """Convert HTML to PDF using WeasyPrint"""
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        font_config = FontConfiguration()
        
        # Create PDF
        HTML(string=html_content).write_pdf(
            output_file,
            font_config=font_config,
            stylesheets=[CSS(string="""
                @page {
                    size: A4;
                    margin: 2cm;
                }
                body {
                    font-size: 11px;
                }
                h1 {
                    page-break-before: always;
                }
                h1:first-child {
                    page-break-before: avoid;
                }
                table {
                    page-break-inside: avoid;
                }
            """)]
        )
        return True
    except Exception as e:
        print(f"WeasyPrint failed: {e}")
        return False

def generate_pdf_from_markdown(md_file, output_file):
    """Generate PDF from markdown file"""
    print(f"Converting {md_file} to PDF...")
    
    # Convert markdown to HTML
    html_content = markdown_to_html(md_file)
    
    # Save HTML file for reference
    html_file = output_file.replace('.pdf', '.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML file saved: {html_file}")
    
    # Convert HTML to PDF
    if html_to_pdf(html_content, output_file):
        print(f"PDF generated successfully: {output_file}")
        return True
    else:
        print("PDF generation failed. HTML file is available for manual conversion.")
        return False

def main():
    """Main function to generate all documentation PDFs"""
    print("🚀 Enhanced Nifty Option Prediction System - Documentation Generator")
    print("=" * 70)
    
    # Install requirements
    install_requirements()
    
    # Documentation files
    docs = [
        ("USER_MANUAL.md", "Enhanced_Nifty_Option_Prediction_System_User_Manual.pdf"),
        ("TECHNICAL_DOCUMENTATION.md", "Enhanced_Nifty_Option_Prediction_System_Technical_Docs.pdf"),
        ("QUICK_REFERENCE.md", "Enhanced_Nifty_Option_Prediction_System_Quick_Reference.pdf")
    ]
    
    success_count = 0
    
    for md_file, pdf_file in docs:
        if os.path.exists(md_file):
            if generate_pdf_from_markdown(md_file, pdf_file):
                success_count += 1
        else:
            print(f"Warning: {md_file} not found!")
    
    print("\n" + "=" * 70)
    print(f"Documentation generation complete!")
    print(f"Successfully generated: {success_count}/{len(docs)} PDF files")
    
    if success_count > 0:
        print("\nGenerated files:")
        for _, pdf_file in docs:
            if os.path.exists(pdf_file):
                print(f"  ✓ {pdf_file}")
    
    print("\n📚 Documentation Summary:")
    print("  • User Manual: Complete system guide for traders")
    print("  • Technical Docs: Developer reference and API documentation")
    print("  • Quick Reference: Trading quick reference card")
    
    print("\n💡 Tips:")
    print("  • Use the Quick Reference card for daily trading")
    print("  • Read the User Manual for comprehensive understanding")
    print("  • Refer to Technical Docs for development and customization")

if __name__ == "__main__":
    main()
