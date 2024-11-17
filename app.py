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
from config import anthropic_api_key, openai_api_key

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
socketio = SocketIO(app)

# Initialize AI clients
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
openai_client = openai.OpenAI(api_key=openai_api_key)

# Sample SMILES strings for random molecule generation
SAMPLE_SMILES = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
    'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Salbutamol
    'CC1=C(C=C(C=C1)O)C(=O)CC2=C(C3=C(C=C2)O)OC(=O)C3',  # Warfarin
    'CN1C=NC2=C1C(=O)NC(=O)N2C',  # Theophylline
    'CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)CC3=CC=C(C=C3)OCC4=CC=CC=C4',  # Cetirizine
    'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Albuterol
]

def make_api_call(client, model_type, prompt):
    try:
        if model_type == "anthropic":
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif model_type == "claude_prompt":
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif model_type == "GPT4o":
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
        else:  # openai
            response = client.chat.completions.create(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'conversation' not in session:
        session['conversation'] = []

    if request.method == 'POST':
        user_input = request.form['user_input']
        model_choice = request.form['model_choice']
        
        try:
            # Get response based on selected model
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
            error_message = f"An error occurred: {str(e)}"
            return jsonify({'response': error_message})

    return render_template('index.html', 
                         conversation=session.get('conversation', []),
                         current_model=request.form.get('model_choice', 'claude_haiku_prompt'))

@app.route('/clear_chat')
def clear_chat():
    session['conversation'] = []
    return '', 204

@socketio.on('get_random_molecule')
def handle_molecule_request():
    try:
        # Select a random SMILES string
        smiles = random.choice(SAMPLE_SMILES)
        
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        # Generate 2D depiction
        img = Draw.MolToImage(mol)
        
        # Convert image to base64 string
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_str = base64.b64encode(img_io.getvalue()).decode()
        
        # Emit the molecule data
        socketio.emit('molecule_response', {'image': img_str, 'smiles': smiles})
        
    except Exception as e:
        print(f"Error generating molecule: {str(e)}")
        socketio.emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
