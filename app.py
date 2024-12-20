from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
import random
import base64
import io
import anthropic
import openai
import numpy as np
import requests
import pandas as pd
import os
from datetime import datetime
from config import anthropic_api_key, openai_api_key, elevenlabs_key
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
socketio = SocketIO(app)

# Initialize API clients
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
openai_client = openai.OpenAI(api_key=openai_api_key)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sample SMILES strings for random molecule generation
SAMPLE_SMILES = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
    'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Salbutamol
    'CC1=C(C=C(C=C1)O)C(=O)CC2=CC=C(C=C2)O',  # Benzestrol
]

# Store uploaded SMILES data
uploaded_smiles = {}

def save_uploaded_file(file):
    """Save uploaded file to the uploads directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath

def make_api_call(client, model_type, message):
    try:
        if model_type == "anthropic":
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                temperature=0.7,
                system="You are a helpful chemistry assistant.",
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
        elif model_type == "claude_prompt":
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                temperature=0.7,
                system="You are a helpful chemistry assistant.",
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
        elif model_type == "GPT4o":
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        else:  # openai
            response = client.chat.completions.create(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def generate_random_molecule():
    try:
        # Select a random SMILES string
        smiles = random.choice(SAMPLE_SMILES)
        
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        return mol
    except Exception as e:
        print(f"Error generating molecule: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def home():
    if 'conversation' not in session:
        session['conversation'] = []
    return render_template('index.html', conversation=session['conversation'])

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Please upload a CSV file'}), 400

        # Save the file
        filepath = save_uploaded_file(file)
        print(f"Saved file to: {filepath}")

        # Read CSV file
        df = pd.read_csv(filepath)
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Look for SMILES column (case-insensitive)
        smiles_col = None
        for col in df.columns:
            if col.upper() == 'SMILES':
                smiles_col = col
                break
                
        if not smiles_col:
            print("No SMILES column found")
            return jsonify({'error': 'CSV file must contain a SMILES column'}), 400

        # Store SMILES data with indices
        global uploaded_smiles
        uploaded_smiles.clear()  # Clear previous data
        
        # Iterate through rows and store valid SMILES
        valid_count = 0
        for idx, row in df.iterrows():
            if pd.notna(row[smiles_col]):
                smiles = str(row[smiles_col]).strip()
                # Validate SMILES string
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
                    uploaded_smiles[valid_count] = smiles
        
        print(f"Stored {len(uploaded_smiles)} valid SMILES entries")
        print(f"First few entries: {dict(list(uploaded_smiles.items())[:3])}")
        
        return jsonify({
            'message': f'Successfully loaded {len(uploaded_smiles)} valid molecules',
            'count': len(uploaded_smiles)
        })

    except Exception as e:
        print(f"Error in upload_csv: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if 'conversation' not in session:
        session['conversation'] = []

    user_input = request.form['user_input']
    model_choice = request.form['model_choice']
    
    try:
        # Check for special commands first
        if user_input.lower().startswith("show molecule"):
            return jsonify({'type': 'molecule'})
        elif user_input.lower().startswith("show plot"):
            return jsonify({'type': 'plot'})
        else:
            # Get AI response based on selected model
            if model_choice == 'openai':
                response_text = make_api_call(openai_client, "openai", user_input)
            elif model_choice == 'claude_haiku_prompt':
                response_text = make_api_call(anthropic_client, "anthropic", user_input)
            elif model_choice == 'claude_prompt':
                response_text = make_api_call(anthropic_client, "claude_prompt", user_input)
            else:  # GPT4o
                response_text = make_api_call(openai_client, "GPT4o", user_input)

        # Add messages to session
        session['conversation'].append(('user', user_input))
        session['conversation'].append(('bot', response_text))
        session.modified = True

        return jsonify({'response': response_text})

    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({'response': f"An error occurred: {str(e)}"})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session['conversation'] = []
    return jsonify({'status': 'success'})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if not audio_file:
            print("Audio file is empty")
            return jsonify({'error': 'Empty audio file'}), 400

        # Save the file temporarily
        temp_path = 'temp_audio.wav'
        audio_file.save(temp_path)
        
        try:
            print("Starting transcription...")
            # Transcribe using OpenAI's Whisper API
            with open(temp_path, 'rb') as audio:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )
            
            print(f"Transcription result: {transcript}")
            return jsonify({'text': transcript})
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return jsonify({'error': f'Transcription error: {str(e)}'}), 500
        finally:
            # Clean up the temporary file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print("Temporary file removed")
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # ElevenLabs API endpoint
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"  # Default voice ID
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Convert audio data to base64
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            return jsonify({'audio': audio_base64})
        else:
            return jsonify({'error': f'ElevenLabs API error: {response.text}'}), response.status_code

    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('trigger_molecule')
def handle_molecule_request():
    try:
        # Generate a random molecule
        mol = generate_random_molecule()
        
        # Calculate molecular weight
        mol_weight = Descriptors.ExactMolWt(mol)
        
        # Convert molecule to image
        img = Draw.MolToImage(mol, size=(300, 300))  # Reduced size
        
        # Save image to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Get SMILES representation
        smiles = Chem.MolToSmiles(mol)
        
        # Encode image to base64
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        
        # Emit the response with molecular weight
        socketio.emit('molecule_response', {
            'image': encoded_img,
            'smiles': smiles,
            'molecular_weight': f"{mol_weight:.2f}"
        })
        
    except Exception as e:
        print(f"Error generating molecule: {str(e)}")
        socketio.emit('error', {'message': str(e)})

@socketio.on('trigger_plot')
def handle_plot_request():
    try:
        # Generate random data
        x = list(range(50))
        y = [random.randint(1, 100) for _ in range(50)]
        
        # Create plot data
        data = [{
            'x': x,
            'y': y,
            'mode': 'lines+markers',
            'type': 'scatter',
            'name': 'Random Data',
            'line': {'color': '#007bff'},
            'marker': {'color': '#007bff'}
        }]
        
        # Create layout
        layout = {
            'title': 'Interactive Random Data Plot',
            'xaxis': {
                'title': 'X-axis',
                'showgrid': True,
                'zeroline': True,
                'gridcolor': '#e0e0e0'
            },
            'yaxis': {
                'title': 'Y-axis',
                'showgrid': True,
                'zeroline': True,
                'gridcolor': '#e0e0e0'
            },
            'showlegend': True,
            'hovermode': 'closest',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {
                'l': 50,
                'r': 30,
                't': 50,
                'b': 50
            }
        }
        
        socketio.emit('graph_response', {'data': data, 'layout': layout})
        
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        socketio.emit('error', {'message': 'Failed to generate plot'})


if __name__ == '__main__':
    # Change to host=0.0.0.0 to allow external access
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)