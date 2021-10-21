from pypointmatcher import pointmatcher as pm, pointmatchersupport as pms
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import sys

def fig_coverage(iteration, value_coverage, path):

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value_coverage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Coverage %", 'font': {'size': 50}},
        #delta = {'reference': 400, 'increasing': {'color': "RebeccaPurple"}},
        gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "green"},
        #'bgcolor': "white",
        'borderwidth': 1,
        'bordercolor': "lightgrey",
        'steps': [
            {'range': [0, 100], 'color': 'lightgrey'},
        #    {'range': [250, 400], 'color': 'black'}
        ],
        #'threshold': {
        #    'line': {'color': "red", 'width': 4},
        #    'thickness': 0.75,
        #    'value': 490}
        }))
    fig.update_layout(paper_bgcolor = "black", font = {'color': "lightgrey", 'family': "Arial"})
    fig.write_image(path)

if(len(sys.argv)!=6):
    print("Error in the number of arguments. "
          "1: input directory name of the reading point-cloud, "
          "2: output directory name for the results, "
          "3: file name of the resulted .png,"
          "4: Threshold int in meter to compute the deviation,"
          "5: Number of point in the ground truth")
    print('Exemple: python3 Compute_deviation_map.py "/home/map/" "/home/deviation/" "deviation_gauge" "1" "1000000"')
    sys.exit()

input_directory = sys.argv[1]
output_base_directory = sys.argv[2]
output_base_file = sys.argv[3]
threshold = sys.argv[4]
total_points_groundtruth = sys.argv[5]   #1 000 000 for darpa subt ground truth

PM = pm.PointMatcher
DP = PM.DataPoints
Parameters = pms.Parametrizable.Parameters

# Load 3D point clouds
txt_folder = Path(input_directory).rglob('map_drift_*')  #input files should for instance map_drift_1.vtk
files = [x for x in txt_folder]
files_sorted = sorted(files, key=lambda x: int((str(x).split('/'))[-1].split('_')[2].split('.')[0]))

iteration = 0
for i in files_sorted:
    number_file = int((str(i).split('/'))[-1].split('_')[2].split('.')[0])
    data = DP(DP.load(str(i)))

    drift_data = data.getDescriptorCopyByName("Drift")
    outliers = np.count_nonzero(drift_data <= int(threshold))
    coverage_percentage = round(outliers/int(total_points_groundtruth)*100)
    if iteration < 10:
        path = f"{output_base_directory}{output_base_file}_{0}{0}{0}{number_file}.png"
    if iteration >= 10 and iteration < 100:
        path = f"{output_base_directory}{output_base_file}_{0}{0}{number_file}.png"
    if iteration >= 100 and iteration < 1000:
        path = f"{output_base_directory}{output_base_file}_{0}{number_file}.png"
    if iteration >= 1000 and iteration < 10000:
        path = f"{output_base_directory}{output_base_file}_{number_file}.png"
    fig_coverage(iteration, coverage_percentage, path)
    iteration = iteration+1

