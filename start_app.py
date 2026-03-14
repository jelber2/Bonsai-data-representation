import os
import sys

results_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/'
)

os.environ['BONSAI_DATA_PATH'] = os.path.abspath(os.path.join(results_folder, 'bonsai_vis_data.hdf'))
os.environ['BONSAI_SETTINGS_PATH'] = os.path.abspath(os.path.join(results_folder, 'bonsai_vis_settings.json'))

if not os.path.exists(os.environ['BONSAI_DATA_PATH']):
    print("ERROR: Preprocessed results not found at:", os.environ['BONSAI_DATA_PATH'])
    sys.exit(1)

if not os.path.exists(os.environ['BONSAI_SETTINGS_PATH']):
    print("ERROR: Settings file not found at:", os.environ['BONSAI_SETTINGS_PATH'])
    sys.exit(1)

import subprocess
subprocess.run([
    'shiny', 'run', 'bonsai_scout/app.py',
    '--port=5000',
    '--host=0.0.0.0',
    '--reload'
])
