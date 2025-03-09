from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from rag import GemRag
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
chatbot = GemRag(api_key=GOOGLE_API_KEY)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            try:
                chatbot.load_pdf(upload_path)
                chatbot.optimize_vector_store()
                os.remove(upload_path)
                return render_template('index.html', success=f"File {filename} processed successfully")
            except Exception as e:
                return render_template('index.html', error=f"Processing failed: {str(e)}")
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = chatbot.chat(user_input)
        return jsonify({
            'response': response,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)