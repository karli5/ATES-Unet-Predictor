import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import joblib

def initialize_scalers(csv_dir, numpy_dir, phase):
    """
    Initializes and calculates minimum and maximum values for vector, heat output, and field data.

    Args:
    csv_dir (str): Directory path containing CSV files.
    numpy_dir (str): Directory path containing numpy files.

    Returns:
    tuple: Min-max values for vector, heat output, and field data.
    """
    min_vector_values, max_vector_values = None, None
    min_heat_out, max_heat_out = None, None
    min_field_value, max_field_value = np.inf, -np.inf
    timestep_start, timestep_end = None
    # Iterate through CSV files to find min and max for vector data
    for file_name in os.listdir(csv_dir):
        phases = [(0,91, "Inj"), (92, 182, "Store"), (183, 273, "Prod"), (274, 364, "Pause")]
        for phase_tuple in phases: 
            if phase in phase_tuple:
                timestep_start, timestep_end =  phase[0], phase[1]
            
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_dir, file_name)
            data = pd.read_csv(file_path)

            filtered_data = data[(data['timestep'] >= timestep_start) & (data['timestep'] <= timestep_end)]

            if phase == "Inj":
                vectors = filtered_data.drop(['timestep', 'Production', 'Injection', 'Pause', 'produced_T'], axis=1)
            if phase == "Store" or phase == "Pause":
                vectors = None
            else: 
                vectors = filtered_data.drop(['timestep', 'Production', 'Injection', 'Pause', 'produced_T', 'temp'], axis=1)

            vectors = vectors.values
            if min_vector_values is None:
                min_vector_values = np.min(vectors, axis=0)
                max_vector_values = np.max(vectors, axis=0)
            else:
                min_vector_values = np.minimum(min_vector_values, np.min(vectors, axis=0))
                max_vector_values = np.maximum(max_vector_values, np.max(vectors, axis=0))

            heat_out_values = data['produced_T'].values
            if min_heat_out is None:
                min_heat_out = np.min(heat_out_values)
                max_heat_out = np.max(heat_out_values)
            else:
                min_heat_out = min(min_heat_out, np.min(heat_out_values))
                max_heat_out = max(max_heat_out, np.max(heat_out_values))  
            
    # Iterate through numpy files to find global min and max for field data
    for numpy_file in os.listdir(numpy_dir):
        if numpy_file.endswith('.npy'):
            field = np.load(os.path.join(numpy_dir, numpy_file))
            min_field_value = min(np.min(field), min_field_value)
            max_field_value = max(np.max(field), max_field_value)

    return (min_vector_values, max_vector_values), (min_heat_out, max_heat_out), (min_field_value, max_field_value)

def create_and_save_scalers(csv_dir, numpy_dir, output_dir):
    """
    Creates and saves MinMaxScaler objects based on data in specified directories.

    Args:
    csv_dir (str): Directory path containing CSV files.
    numpy_dir (str): Directory path containing numpy files.
    output_dir (str): Directory path to save the scaler files.
    """
    for phase in ["Inj", "Store", "Prod", "Pause"]:
        # Calculate the min and max values
        vector_min_max, heat_out_min_max, field_min_max = initialize_scalers(csv_dir, numpy_dir, phase)

        # Create and train scalers
        vector_scaler = MinMaxScaler()
        vector_scaler.fit([vector_min_max[0], vector_min_max[1]])
        
        heat_out_scaler = MinMaxScaler()
        heat_out_scaler.fit([[heat_out_min_max[0]], [heat_out_min_max[1]]])

        field_scaler = MinMaxScaler()
        field_scaler.fit([[field_min_max[0]], [field_min_max[1]]])

        # Save the scalers
        dump(vector_scaler, os.path.join(output_dir, 'vector_scaler_{phase}.joblib'))
        dump(heat_out_scaler, os.path.join(output_dir, 'heat_out_scaler_{phase}.joblib'))
        dump(field_scaler, os.path.join(output_dir, 'field_scaler_{phase}.joblib'))

        print("Scalers saved successfully.")
    


if __name__ == "__main__":
    csv_directory = 'Data_Vector_Scalar'
    numpy_directory = 'TemperatureFields/NumpyArrays'
    output_directory = 'Scalers'

    create_and_save_scalers(csv_directory, numpy_directory, output_directory)