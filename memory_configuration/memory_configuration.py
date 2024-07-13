from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    num_ues = int(request.form['num_ues'])
    memory_needed = get_memory_needed(num_ues)
    return render_template('result.html', memory_needed=memory_needed, num_ues=num_ues)

def convert_to_KB(value):
    return round(value * 1024, 3)

def get_memory_needed(num_ues):
    url = "http://127.0.0.1:5001/api/mempool_multipliers"
    response = requests.get(url)
    if response.status_code == 200:
        mempool_multipliers = response.json()
        memory_needed = {}
        for memtype, values in mempool_multipliers.items():
            total_memory = {}
            for ue, multiplier in values.items():
                if "Static memory" in ue:
                    memory_in_mb = round(multiplier, 3)  # Extract static memory value
                else:
                    memory_in_mb = round(multiplier * num_ues, 3)  # Calculate memory based on multiplier
                memory_in_kb = convert_to_KB(memory_in_mb)
                total_memory[ue.replace("P_amf", "").replace("Per UE ", "")] = (str(memory_in_mb) + " MB", str(memory_in_kb) + " KB")
            memory_needed[memtype] = total_memory
        return memory_needed
    else:
        return None


if __name__ == "__main__":
    app.run(debug=True)
