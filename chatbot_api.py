"""
🤖 TRINETRA Chatbot API
Flask backend for Groq-powered chatbot
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Groq API configuration
# Use environment variable if available, otherwise use the provided key
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Updated to use llama-3.3-70b-versatile (llama-3.1-70b-versatile was decommissioned)
# You can override this via environment variable: GROQ_MODEL
# Other available models: llama-3.1-70b-instant, llama-3.1-8b-instant, mixtral-8x7b-32768
MODEL = os.environ.get('GROQ_MODEL', "llama-3.3-70b-versatile")

def get_system_prompt(selected_object=None):
    """Generate system prompt based on context"""
    base_prompt = """You are TRINETRA, an AI assistant for a space object tracking and visualization system. 
TRINETRA provides real-time tracking of satellites, rockets, debris, and other space objects orbiting Earth.

You can help users with:
- General questions about space objects, satellites, orbits, and space technology
- Questions about specific objects when they are selected in the visualization
- Information about orbital mechanics, space debris, satellite operations
- Conjunction assessments and collision risks
- Space mission information and satellite constellations

Be helpful, accurate, and concise. If you don't know something, say so."""
    
    if selected_object:
        base_prompt += f"\n\nCurrent context: The user has selected object '{selected_object}'. You can provide specific information about this object when asked."
    
    return base_prompt

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot messages"""
    try:
        data = request.json
        message = data.get('message', '')
        selected_object = data.get('selectedObject', None)
        conversation_history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Prepare messages for Groq API
        messages = [
            {
                "role": "system",
                "content": get_system_prompt(selected_object)
            }
        ]
        
        # Add conversation history (last 10 messages to avoid token limits)
        for msg in conversation_history[-10:]:
            messages.append({
                "role": msg.get('role', 'user'),
                "content": msg.get('content', '')
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Call Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            error_msg = f"Groq API error: {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', error_msg)
            except:
                pass
            return jsonify({'error': error_msg}), response.status_code
        
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        
        return jsonify({
            'response': assistant_message,
            'model': MODEL
        })
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout. Please try again.'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'TRINETRA Chatbot API'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
