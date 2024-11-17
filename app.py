from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import random
import base64
import io
import anthropic
import openai
import numpy as np
from config import anthropic_api_key, openai_api_key

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
socketio = SocketIO(app)

# Initialize API clients
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
openai_client = openai.OpenAI(api_key=openai_api_key)

# Sample SMILES strings for random molecule generation
SAMPLE_SMILES = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
    'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Salbutamol
    'CC1=C(C=C(C=C1)O)C(=O)CC2=CC=C(C=C2)O',  # Benzestrol
]

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

@app.route('/chat', methods=['POST'])
def chat():
    if 'conversation' not in session:
        session['conversation'] = []

    user_input = request.form['user_input']
    model_choice = request.form['model_choice']
    
    try:
        # Check for special commands first
        if user_input.lower() == "show molecule":
            response_text = "Here's a random molecule for you!"
            # Generate and emit molecule separately
            socketio.emit('trigger_molecule')
        elif user_input.lower() == "show plot":
            response_text = "Here's an interactive plot for you!"
            # Generate and emit plot separately
            socketio.emit('trigger_plot')
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
        return jsonify({'response': f"An error occurred: {str(e)}"})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session['conversation'] = []
    return jsonify({'status': 'success'})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        
        # Save the file temporarily
        temp_path = 'temp_audio.wav'
        audio_file.save(temp_path)
        
        try:
            # Transcribe using OpenAI's Whisper API
            with open(temp_path, 'rb') as audio:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )
                
            return jsonify({'text': response})
            
        finally:
            # Clean up the temporary file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
    except Exception as e:
        print(f"Transcription error: {str(e)}")
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
        # Generate random molecule
        mol = generate_random_molecule()
        if mol is not None:
            # Convert molecule to image
            img = Draw.MolToImage(mol)
            
            # Convert image to base64
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Get SMILES representation
            smiles = Chem.MolToSmiles(mol)
            
            # Emit the molecule data
            socketio.emit('molecule_response', {
                'image': img_str,
                'smiles': smiles
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
            'name': 'Random Data'
        }]
        
        # Create layout
        layout = {
            'title': 'Interactive Random Data Plot',
            'xaxis': {
                'title': 'X-axis',
                'showgrid': True,
                'zeroline': True
            },
            'yaxis': {
                'title': 'Y-axis',
                'showgrid': True,
                'zeroline': True
            },
            'showlegend': True,
            'hovermode': 'closest',
            'width': 560,  # Adjusted to fit container
            'height': 380, # Adjusted to fit container
            'margin': {    # Added margins for better fit
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
    socketio.run(app, debug=True)
