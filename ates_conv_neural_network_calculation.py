from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Dense, Reshape, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from joblib import load
from data_generator_ates import TempFieldDataGeneratorInjection, TempFieldDataGeneratorStorage, TempFieldDataGeneratorProduction, ScalarDataGeneratorProduction, TempFieldDataGeneratorPause
import numpy as np
import pandas as pd
import os


# Define paths
csv_dir = 'Daten_Vector_Scalar'
numpy_dir = 'TemperatureFields'
scaler_dir = 'Scalers'
model_dir = 'AI_Models'

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)

for phase in ["Inj", "Store", "Prod", "Prod_Scalar", "Pause"]:

    # Load scalers
    if phase == "Prod_scalar":
        vector_scaler = load(os.path.join(scaler_dir, 'vector_scaler_Prod.joblib'))
        heat_out_scaler = load(os.path.join(scaler_dir, 'heat_out_scaler_Prod.joblib'))
        field_scaler = load(os.path.join(scaler_dir, 'field_scaler_Prod.joblib'))        
    else: 
        vector_scaler = load(os.path.join(scaler_dir, 'vector_scaler_{phase}.joblib'))
        heat_out_scaler = load(os.path.join(scaler_dir, 'heat_out_scaler_{phase}.joblib'))
        field_scaler = load(os.path.join(scaler_dir, 'field_scaler_{phase}.joblib'))

    model_checkpoint_path = os.path.join(model_dir, 'model_{phase}.keras')

    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_loss', 
        verbose=1,  
        save_best_only=True, 
        mode='min'  
    )

    def create_model(phase):
        """
        Create and return a Keras model for temperature field prediction.
        """
        # Vector input
        if phase == "Inj": 
            input_vector = Input(shape=(2,), name='input_vector')
        if phase == "Store" or phase == "Pause":
            input_vector = None
        else: 
            input_vector = Input(shape=(1,), name='input_vector')

        # Temperature field input
        input_field = Input(shape=(256, 256, 1), name='input_field')

        # Project each of the 5 values onto a separate 256x256 field
        if phase == "Prod" or phase == "Inj" or phase == "Prod_Scalar":
            vector_projection = Dense(256*256*2)(input_vector) 
            vector_projection = Reshape((256, 256, 2))(vector_projection)  
            combined_input = Concatenate(axis=-1)([input_field, vector_projection])
            x = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(combined_input)

        else: 
        # Concatenate the upscaled vector field and the temperature field
            x = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(input_field)

        # Encoder layers
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x1 = x 

        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x2 = x

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x3 = x 

        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x4 = x  

        # Residual Blocks
        for _ in range(6):
            res_x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            res_x = BatchNormalization()(res_x)
            res_x = Activation('relu')(res_x)
            res_x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(res_x)
            res_x = BatchNormalization()(res_x)
            x = Activation('relu')(x + res_x) 

        # Decoder layers
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, x3])

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, x2])

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, x1])

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if phase == "Inj":
            # Output layer for the temperature field
            output_field = Conv2D(1, (3, 3), padding='same', name='output_field')(x)
            # Assemble the model
            model = Model(inputs=[input_vector, input_field], outputs=[output_field])

        if phase == "Store" or phase == "Pause": 
            # Output layer for the temperature field
            output_field = Conv2D(1, (3, 3), padding='same', name='output_field')(x)
            # Assemble the model
            model = Model(inputs=[input_field], outputs=[output_field])

        if phase == "Prod": 
            # Output layer for the temperature field
            output_field = Conv2D(1, (3, 3), padding='same', name='output_field')(x)
            # Assemble the model
            model = Model(inputs=[input_vector, input_field], outputs=[output_field])            
        if phase ==  "Prod_Scalar":
            # Flat vector and field to coombine them
            vector_flat = Flatten()(input_vector)
            field_flat = Flatten()(x)
            # Combine flattened layers
            combined = Concatenate()([vector_flat, field_flat])
            # Output layer for scalar
            output_scalar = Dense(1, activation='linear', name='output_scalar')(combined)
            # Assemble the model
            model = Model(inputs=[input_vector, input_field], outputs=[output_scalar])

        return model

    def extract_combinations(csv_dir, phase):
        """
        Extracts unique combinations of fluid mass, temperature, and timestep from CSV files.
        """
        if phase == "Inj":
            timestep_start = 0
            timestep_end = 91
        if phase == "Store":
            timestep_start = 92
            timestep_end = 182
        if phase == "Prod" or phase == "Prod_Scalar": 
            timestep_start = 183
            timestep_end = 273
        else: 
            timestep_start = 274
            timestep_end = 364            
        combinations = set()
        for csv_file in os.listdir(csv_dir):
            if csv_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(csv_dir, csv_file))
                for _, row in df.iterrows():
                    fluid_mass = str(float(row['fluid_mass']))
                    temp = str(int(row['temp']))
                    timestep = str(int(row['timestep']))
                    if (int(timestep) >= timestep_start) and (int(timestep) < timestep_end):
                        combinations.add((fluid_mass, temp, timestep))
        return list(combinations)


    # Create the model
    model = create_model(phase)
    if phase != "Prod_scalar":
        model.compile(optimizer='adam', loss={'output_field': 'mean_squared_error'}, metrics={'output_field': 'mean_absolute_error'})
    else: 
        model.compile(optimizer='adam', loss={'output_scalar': 'mean_squared_error'}, metrics={'output_scalar': 'mean_absolute_error'})

    model.summary()

    # Prepare data generators
    combinations_list = extract_combinations(csv_dir, phase)
    train_combinations, test_combinations = train_test_split(combinations_list, test_size=0.2, random_state=42)
    if phase == "Inj": 
        train_generator = TempFieldDataGeneratorInjection(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, train_combinations, batch_size=32, shuffle=True)
        test_generator = TempFieldDataGeneratorInjection(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, test_combinations, batch_size=32, shuffle=False)
    if phase == "Store":
        train_generator = TempFieldDataGeneratorStorage(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, train_combinations, batch_size=32, shuffle=True)
        test_generator = TempFieldDataGeneratorStorage(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, test_combinations, batch_size=32, shuffle=False)        
    if phase == "Prod":
        train_generator = TempFieldDataGeneratorProduction(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, train_combinations, batch_size=32, shuffle=True)
        test_generator = TempFieldDataGeneratorProduction(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, test_combinations, batch_size=32, shuffle=False)
    if phase == "Prod_Scalar":
        train_generator = ScalarDataGeneratorProduction(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, train_combinations, batch_size=32, shuffle=True)
        test_generator = ScalarDataGeneratorProduction(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, test_combinations, batch_size=32, shuffle=False)
    if phase == "Pause": 
        train_generator = TempFieldDataGeneratorPause(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, train_combinations, batch_size=32, shuffle=True)
        test_generator = TempFieldDataGeneratorPause(csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, test_combinations, batch_size=32, shuffle=False)
    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x=train_generator, steps_per_epoch=len(train_generator), epochs=200, validation_data=test_generator, validation_steps=len(test_generator), callbacks=[early_stop, model_checkpoint_callback])

    # Evaluate the model
    results = model.evaluate(x=test_generator, steps=len(test_generator))
    print("Test Loss, Test Accuracy:", results)
