from flask import Flask, request, jsonify
from queue import Queue

app = Flask(__name__)

# Initialize a queue to store the strings
audio_queue = Queue()
audio_timestamp_queue = Queue()
vision_queue = Queue()
vision_timestamp_queue = Queue()

@app.route('/audio/post_wav', methods=['POST'])
def post_wav():
    data = request.get_json()
    if not data or 'audio_string' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    audio_queue.put(data['audio_string'])
    audio_timestamp_queue.put(data['timestamp'])
    return jsonify({'message': 'Audio string added to queue'}), 200

@app.route('/audio/get_wav', methods=['GET'])
def get_wav():
    if audio_queue.empty():
        return jsonify({'info': 'Queue is empty'}), 200
    
    audio_string = audio_queue.get()
    timestamp = audio_timestamp_queue.get()
    return jsonify({'info': audio_string, 'timestamp' : timestamp}), 200

@app.route('/vision/post_jpg', methods=['POST'])
def post_jpg():
    data = request.get_json()
    if not data or 'vision_string' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    vision_queue.put(data['vision_string'])
    vision_timestamp_queue.put(data['timestamp'])
    return jsonify({'message': 'Vision string added to queue'}), 200

@app.route('/vision/get_jpg', methods=['GET'])
def get_jpg():
    if vision_queue.empty():
        return jsonify({'info': 'Queue is empty'}), 200
    
    vision_string = vision_queue.get()
    timestamp = vision_timestamp_queue.get()
    return jsonify({'info': vision_string, 'timestamp' : timestamp}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
