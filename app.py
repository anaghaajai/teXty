import flask
import pytesseract
from gingerit.gingerit import GingerIt
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from reportlab.pdfgen import canvas

app = flask.Flask(__name__)

def extract_text_from_image(image_path):
    # Use OCR to extract text from the handwritten images
    text = pytesseract.image_to_string(image_path)
    return text

def correct_grammar(text):
    parser = GingerIt()
    result = parser.parse(text)
    corrected_text = result['result']
    return corrected_text

def calculate_accuracy(reference_text, corrected_text):
    reference_words = reference_text.split()
    corrected_words = corrected_text.split()

    num_correct = 0
    num_total = len(reference_words)

    for reference_word, corrected_word in zip(reference_words, corrected_words):
        if reference_word.lower() == corrected_word.lower():
            num_correct += 1

    accuracy = (num_correct / num_total) * 100
    return accuracy

def self_feeding_training(image_path, reference_text, num_iterations=10):
    global corrected_text
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    optimizer = AdamW(model.parameters(), lr=1e-5)  # Initialize optimizer

    for i in range(num_iterations):
        # Extract text from the handwritten image
        handwritten_text = extract_text_from_image(image_path)

        # Correct grammar errors in the extracted text
        corrected_text = correct_grammar(handwritten_text)

        # Train the model on the corrected text
        inputs = tokenizer.encode(corrected_text, return_tensors="pt")

        if inputs.shape[1] > 0:  # Check if the tensor has non-zero dimensions
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()

            # Update the model parameters
            optimizer.step()
            optimizer.zero_grad()

    accuracy = calculate_accuracy(reference_text, corrected_text)
    print("Accuracy: {:.2f}%".format(accuracy))  # Print accuracy on the terminal
    return corrected_text

def generate_pdf(text):
    pdf_path = "static/output/output.pdf"  # Path to save the generated PDF
    c = canvas.Canvas(pdf_path)

    # Set the font and size
    c.setFont("Helvetica", 12)

    # Split the text into lines
    lines = text.split("\n")

    # Write each line to the PDF
    y = 700  # Initial y-coordinate
    line_height = 15  # Height of each line

    for line in lines:
        c.drawString(50, y, line)
        y -= line_height

    # Save and close the PDF
    c.save()

    return pdf_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'POST':
        image_file = flask.request.files['image']
        image_path = "static/images/sample.jpg" # Update the file path here
        image_file.save(image_path)

        reference_text = "Sample data to check the working of back-end program"  # Update with your reference text

        converted_text = self_feeding_training(image_path, reference_text)
        pdf_path = generate_pdf(converted_text)
        return flask.render_template('result.html', converted_text=converted_text, pdf_path=pdf_path)
    return flask.render_template('index.html')

@app.route('/download', methods=['GET'])
def download():
    pdf_path = flask.request.args.get('pdf_path')
    return flask.send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run()
