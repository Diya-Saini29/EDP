#!/bin/bash
echo "🚀 Starting Jetson Nano Edge Deployment..."

sudo nvpmodel -m 0
sudo jetson_clocks

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r Requirements.txt

if [ ! -f "traffic_model_96.engine" ]; then
    echo "🔧 Running TensorRT optimization..."
    python3 optimize.py
fi

echo "📊 Running benchmark..."
python3 benchmark.py

# THIS IS THE CHANGE - using headless version
echo "🎥 Recording demo video..."
python3 main_headless.py

echo "✅ Done! Video saved as jetson_demo.mp4"