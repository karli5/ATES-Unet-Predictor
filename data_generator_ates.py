from keras.utils import Sequence
import numpy as np
import pandas as pd
import os

class BaseDataGenerator(Sequence):
    def __init__(self, csv_dir, numpy_dir, vector_scaler, heat_out_scaler, field_scaler, combinations, batch_size=4, shuffle=True):
        self.csv_dir = csv_dir
        self.numpy_dir = numpy_dir
        self.vector_scaler = vector_scaler
        self.heat_out_scaler = heat_out_scaler
        self.field_scaler = field_scaler
        self.combinations = combinations
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.combinations) / self.batch_size))

    def __getitem__(self, index):
        batch_combinations = self.combinations[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_combinations)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.combinations)

    def __data_generation(self, batch_combinations):
        raise NotImplementedError("This method needs to be implemented in subclasses.")

# Beispiel für eine spezialisierte Klasse, die von BaseDataGenerator erbt
class TempFieldDataGeneratorInjection(BaseDataGenerator):
    def __data_generation(self, batch_combinations):
        """
        Generates data for a batch of combinations.

        Args:
        batch_combinations (list of tuples): List containing combinations of mass, temperature, and timestep.

        Returns:
        list: Batch of scaled input and label data.
        """
        X_vector, X_field, y_field_next = [], [], []

        for mass, temp, timestep in batch_combinations:
            # Format mass in scientific notation
            mass_formatted = "{:.0e}".format(float(mass)).replace("e+0", "E").replace("e+", "E")
            csv_file = f"vector_scalar_fluid_mass_{mass_formatted}_temp_{temp}.csv"

            # Check if the CSV file exists
            if not os.path.exists(os.path.join(self.csv_dir, csv_file)):
                continue

            df = pd.read_csv(os.path.join(self.csv_dir, csv_file))

            # Check if the current timestep is in the dataframe
            if int(timestep) in df['timestep'].values:
                vector_data = df.loc[df['timestep'] == int(timestep), ['fluid_mass', 'temp']].values[0]
                numpy_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{timestep}.npy"
                numpy_next_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{int(timestep)+1}.npy"

                field = np.load(os.path.join(self.numpy_dir, numpy_file))

                # Check if the next timestep numpy file exists
                if os.path.exists(os.path.join(self.numpy_dir, numpy_next_file)):
                    field_next = np.load(os.path.join(self.numpy_dir, numpy_next_file))
                else:
                    # Skip if the next field does not exist
                    continue  

                X_vector.append(vector_data)
                X_field.append(field)
                y_field_next.append(field_next)

        # Scaling the data
        X_vector = self.vector_scaler.transform(np.array(X_vector))
        X_field = np.array(X_field).reshape(-1, 256, 256, 1) 
        X_field = self.field_scaler.transform(X_field.reshape(-1, 1)).reshape(-1, 256, 256, 1) 
        y_field_next = np.array(y_field_next).reshape(-1, 256, 256, 1)
        y_field_next = self.field_scaler.transform(y_field_next.reshape(-1, 1)).reshape(-1, 256, 256, 1)  
        return [X_vector, X_field], [y_field_next] 


class TempFieldDataGeneratorStorage(BaseDataGenerator):
    def __data_generation(self, batch_combinations):
        """
        Generates data for a batch of combinations.

        Args:
        batch_combinations (list of tuples): List containing combinations of mass, temperature, and timestep.

        Returns:
        list: Batch of scaled input and label data.
        """
        X_field, y_field_next = [], []

        for mass, temp, timestep in batch_combinations:
            # Format mass in scientific notation
            mass_formatted = "{:.0e}".format(float(mass)).replace("e+0", "E").replace("e+", "E")
            csv_file = f"vector_scalar_fluid_mass_{mass_formatted}_temp_{temp}.csv"

            # Check if the CSV file exists
            if not os.path.exists(os.path.join(self.csv_dir, csv_file)):
                continue

            df = pd.read_csv(os.path.join(self.csv_dir, csv_file))

            # Check if the current timestep is in the dataframe
            if int(timestep) in df['timestep'].values:
                numpy_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{timestep}.npy"
                numpy_next_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{int(timestep)+1}.npy"

                field = np.load(os.path.join(self.numpy_dir, numpy_file))

                # Check if the next timestep numpy file exists
                if os.path.exists(os.path.join(self.numpy_dir, numpy_next_file)):
                    field_next = np.load(os.path.join(self.numpy_dir, numpy_next_file))
                else:
                    # Skip if the next field does not exist
                    continue  

                X_field.append(field)
                y_field_next.append(field_next)

        # Scaling the data
        X_field = np.array(X_field).reshape(-1, 256, 256, 1) 
        X_field = self.field_scaler.transform(X_field.reshape(-1, 1)).reshape(-1, 256, 256, 1)
        y_field_next = np.array(y_field_next).reshape(-1, 256, 256, 1) 
        y_field_next = self.field_scaler.transform(y_field_next.reshape(-1, 1)).reshape(-1, 256, 256, 1)
        return [X_field], [y_field_next] 
    
class TempFieldDataGeneratorProduction(BaseDataGenerator):
    def __data_generation(self, batch_combinations):
        """
        Generates data for a batch of combinations.

        Args:
        batch_combinations (list of tuples): List containing combinations of mass, temperature, and timestep.

        Returns:
        list: Batch of scaled input and label data.
        """
        X_vector, X_field, y_field_next = [], [], [], []

        for mass, temp, timestep in batch_combinations:
            # Format mass in scientific notation
            mass_formatted = "{:.0e}".format(float(mass)).replace("e+0", "E").replace("e+", "E")
            csv_file = f"vector_scalar_fluid_mass_{mass_formatted}_temp_{temp}.csv"

            # Check if the CSV file exists
            if not os.path.exists(os.path.join(self.csv_dir, csv_file)):
                continue

            df = pd.read_csv(os.path.join(self.csv_dir, csv_file))

            # Check if the current timestep is in the dataframe
            if int(timestep) in df['timestep'].values:
                vector_data = df.loc[df['timestep'] == int(timestep), ['fluid_mass']].values[0]
                numpy_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{timestep}.npy"
                numpy_next_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{int(timestep)+1}.npy"

                field = np.load(os.path.join(self.numpy_dir, numpy_file))

                # Check if the next timestep numpy file exists
                if os.path.exists(os.path.join(self.numpy_dir, numpy_next_file)):
                    field_next = np.load(os.path.join(self.numpy_dir, numpy_next_file))
                else:
                    # Skip if the next field does not exist
                    continue  

                X_vector.append(vector_data)
                X_field.append(field)
                y_field_next.append(field_next)

        # Scaling the data
        X_vector = self.vector_scaler.transform(np.array(X_vector))
        X_field = np.array(X_field).reshape(-1, 256, 256, 1)  
        X_field = self.field_scaler.transform(X_field.reshape(-1, 1)).reshape(-1, 256, 256, 1) 
        y_field_next = np.array(y_field_next).reshape(-1, 256, 256, 1) 
        y_field_next = self.field_scaler.transform(y_field_next.reshape(-1, 1)).reshape(-1, 256, 256, 1) 

        return [X_vector, X_field], [y_field_next] 
    
class ScalarDataGeneratorProduction(BaseDataGenerator):
    def __data_generation(self, batch_combinations):
        """
        Generates data for a batch of combinations.

        Args:
        batch_combinations (list of tuples): List containing combinations of mass, temperature, and timestep.

        Returns:
        list: Batch of scaled input and label data.
        """
        X_vector, X_field, y_scalar = [], [], []

        for mass, temp, timestep in batch_combinations:
            # Format mass in scientific notation
            mass_formatted = "{:.0e}".format(float(mass)).replace("e+0", "E").replace("e+", "E")
            csv_file = f"vector_scalar_fluid_mass_{mass_formatted}_temp_{temp}.csv"

            # Check if the CSV file exists
            if not os.path.exists(os.path.join(self.csv_dir, csv_file)):
                continue

            df = pd.read_csv(os.path.join(self.csv_dir, csv_file))

            # Check if the current timestep is in the dataframe
            if int(timestep) in df['timestep'].values:
                vector_data = df.loc[df['timestep'] == int(timestep), ['fluid_mass']].values[0]
                scalar_output = df.loc[df['timestep'] == int(timestep), 'produced_T'].values[0]
                numpy_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{timestep}.npy"
                field = np.load(os.path.join(self.numpy_dir, numpy_file))


                X_vector.append(vector_data)
                X_field.append(field)
                y_scalar.append(scalar_output)

        # Scaling the data
        X_vector = self.vector_scaler.transform(np.array(X_vector))
        X_field = np.array(X_field).reshape(-1, 256, 256, 1)  
        X_field = self.field_scaler.transform(X_field.reshape(-1, 1)).reshape(-1, 256, 256, 1) 
        y_scalar_scaled = self.heat_out_scaler.transform(np.array(y_scalar).reshape(-1, 1))
        return [X_vector, X_field], [y_scalar_scaled]
    
class TempFieldDataGeneratorPause(BaseDataGenerator):
    def __data_generation(self, batch_combinations):
        """
        Generates data for a batch of combinations.

        Args:
        batch_combinations (list of tuples): List containing combinations of mass, temperature, and timestep.

        Returns:
        list: Batch of scaled input and label data.
        """
        X_field, y_field_next = [], []

        for mass, temp, timestep in batch_combinations:
            # Format mass in scientific notation
            mass_formatted = "{:.0e}".format(float(mass)).replace("e+0", "E").replace("e+", "E")
            csv_file = f"vector_scalar_fluid_mass_{mass_formatted}_temp_{temp}.csv"

            # Check if the CSV file exists
            if not os.path.exists(os.path.join(self.csv_dir, csv_file)):
                continue

            df = pd.read_csv(os.path.join(self.csv_dir, csv_file))

            # Check if the current timestep is in the dataframe
            if int(timestep) in df['timestep'].values:
                numpy_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{timestep}.npy"
                numpy_next_file = f"temperature_distribution_mass{mass_formatted}_temp{temp}_timestep{int(timestep)+1}.npy"

                field = np.load(os.path.join(self.numpy_dir, numpy_file))

                # Check if the next timestep numpy file exists
                if os.path.exists(os.path.join(self.numpy_dir, numpy_next_file)):
                    field_next = np.load(os.path.join(self.numpy_dir, numpy_next_file))
                else:
                    # Skip if the next field does not exist
                    continue  

                X_field.append(field)
                y_field_next.append(field_next)

        # Scaling the data
        X_field = np.array(X_field).reshape(-1, 256, 256, 1) 
        X_field = self.field_scaler.transform(X_field.reshape(-1, 1)).reshape(-1, 256, 256, 1) 
        return [X_field], [y_field_next] 