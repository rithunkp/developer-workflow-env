import pypdf

def extract_pdf(filename, output_name):
    try:
        reader = pypdf.PdfReader(filename)
        text = [page.extract_text() for page in reader.pages]
        with open(output_name, 'w', encoding='utf-8') as f:
            f.write("\n".join(text))
        print(f"Successfully extracted {filename} to {output_name}")
    except Exception as e:
        print(f"Error extracting {filename}: {e}")

extract_pdf("Meta OpenEnv Hackathon_ Guidelines.pdf", "guidelines.txt")
extract_pdf("Meta Hackathon Problem Statement.pdf", "problem_statement.txt")
