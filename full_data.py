import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Path to your CSV
csv_path = r"C:\Users\Shivam Taneja\Desktop\Projects\chatbot\arena_car_specs.csv"
df = pd.read_csv(csv_path)

# Path to PDF
pdf_path = r"C:\Users\Shivam Taneja\Desktop\Projects\chatbot\arena_car_specs.pdf"

# PDF Setup
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
normal_style = styles["Normal"]

elements = []

# Title Page
elements.append(Paragraph("Maruti Suzuki Arena - Car Specifications Report", styles["Title"]))
elements.append(Spacer(1, 20))
elements.append(PageBreak())

# One car per page
for _, row in df.iterrows():
    car_name = row.get("Model (Header)", "Unknown Model")

    # Heading
    elements.append(Paragraph(car_name, styles["Heading2"]))
    elements.append(Spacer(1, 10))

    # Build table with Paragraphs (so text wraps)
    data = []
    for col in df.columns:
        val = str(row[col])
        if val == "nan":
            continue
        data.append([Paragraph(str(col), normal_style), Paragraph(val, normal_style)])

    table = Table(data, colWidths=[120, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))

    elements.append(table)
    elements.append(PageBreak())

# Build PDF
doc.build(elements)
print(f"✅ PDF generated successfully: {pdf_path}")
