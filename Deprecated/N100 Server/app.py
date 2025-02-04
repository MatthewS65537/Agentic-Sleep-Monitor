from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_request():
    # Get the raw bytes from the request
    raw_data = request.get_data()
    
    # Decode the bytes (assuming UTF-8 encoding)
    # decoded_data = raw_data.decode('utf-8')
    
    # If you're not sure about the encoding, you can try:
    decoded_data = raw_data.decode('utf-8', errors='replace')
    
    # Process the decoded data
    print(decoded_data)
    
    return "Data received and decoded"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345, debug=True)