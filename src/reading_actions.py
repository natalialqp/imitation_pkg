import pandas as pd
import numpy as np

class SignalAnalyzer:
    def __init__(self, csv_file_path):
        self.df = pd.read_csv("GMM_learned_actions/" + csv_file_path)
        self.time_column = self.df['time']
        self.signal_columns = self.df.drop(columns=['time'])
        self.change_points = []

    def analyze_signals(self):
        # Initialize previous_slopes with the slopes for each signal
        previous_slopes = {signal_name: np.sign(self.signal_columns[signal_name].values[1] - self.signal_columns[signal_name].values[0]) for signal_name in self.signal_columns.columns}

        # for time, values in zip(self.time_column, self.signal_columns.values):
        for i in range(1, len(self.time_column)):
            current_slopes = {signal_name: np.sign(self.signal_columns[signal_name].values[i] - self.signal_columns[signal_name].values[i-1]) for signal_name in self.signal_columns.columns}
            # Check if any signal's slope changes at this time point
            if any(current_slopes[signal_name] != previous_slopes[signal_name] for signal_name in self.signal_columns.columns):
                change_point = {'time': self.time_column.iloc[i]}
                change_point.update({self.signal_columns.columns[k]: value for k, value in enumerate(self.signal_columns.iloc[i])})
                self.change_points.append(change_point)
            previous_slopes = current_slopes
        last_time = self.time_column.iloc[-1]
        last_values = self.signal_columns.iloc[-1]
        last_change_point = {'time': last_time, **{self.signal_columns.columns[i]: value for i, value in enumerate(last_values)}}
        self.change_points.append(last_change_point)

    def save_results(self, name):
        output_file='GMM_learned_actions/' + name + '_for_execution' + '.csv'
        change_points_df = pd.DataFrame(self.change_points)
        change_points_df.to_csv(output_file, index=False)

# Example usage:
name = 'qt_teapot_left'
csv_file_path = name + '.csv'  # Replace with the actual path to your CSV file
analyzer = SignalAnalyzer(csv_file_path)
analyzer.analyze_signals()
analyzer.save_results(name)