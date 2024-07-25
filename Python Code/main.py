# main.py

from data_collection.collect_data_2 import CsvCollector
import json
from datetime import datetime
import sys
import pandas as pd
from static_prediction.split_and_predict import Predictor
import seaborn as sns
import matplotlib.pyplot as plt
import threading
# import dcor
# from static_prediction.check_corelation import CorrelationChecker

def collect_data_for_component(start_time, end_time, registration_number, ops_per_second, component):
    with open("../processing_configuration.json", "r") as config_file:
        config = json.load(config_file)
    
    config['start_time'] = datetime.fromisoformat(start_time)
    config['end_time'] = datetime.fromisoformat(end_time)
    config['component'] = component
    print(config['start_time'])
    collector = CsvCollector(config['start_time'], config['end_time'], config, registration_number, ops_per_second, component)
    collector.collect_csv_data()

def main(start_time, end_time, registration_number, ops_per_second):
    with open("../processing_configuration.json", "r") as config_file:
        config = json.load(config_file)

    components = config['component']
    if isinstance(components, list):
        threads = []
        for component in components:
            thread = threading.Thread(target=collect_data_for_component, args=(start_time, end_time, registration_number, ops_per_second, component))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    else:
        collect_data_for_component(start_time, end_time, registration_number, ops_per_second, components)

    # dataset_name = "amf_1000_1_set.csv"
    # df = pd.read_csv(f"data/{dataset_name}")
    # predictor = Predictor(df.apply(pd.to_numeric, errors='coerce'), config, dataset_name)
    # predictor.split()
    # predictor.predict_on_test_data()  # Example usage of predict_on_test_data method

    # per_ue_amf_needed_df = pd.read_csv("data/per_ue_amf_needed.csv")
    # print(per_ue_amf_needed_df)
    # df_1 = df.iloc[:100]
    # df_2 = pd.read_csv("data/smf_1000_5_set.csv")
    # df_3 = pd.read_csv("data/smf_1000_7_set_1.csv")

    # df_4 = df.iloc[100:200]

    #df_5 = df.iloc[:100]

    # if config['check_corelation']:
    #     checker = CorrelationChecker(df)
    #     #checker.compute_correlations()
    #     checker.exhaustive_feature_selection("phoenix_memory_used_cm_sessionP_smf")
        # checker.fisher_score_feature_selection("phoenix_memory_used_cm_sessionP_smf")
        #checker.chi_squared_feture_selection('phoenix_memory_used_cm_sessionP_smf')
    #df = pd.concat([df_2, df])



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py <start_time> <end_time> <registration_number> <ops_per_second>")
        sys.exit(1)
    
    start_time = sys.argv[1]
    end_time = sys.argv[2]
    registration_number = sys.argv[3]
    ops_per_second = sys.argv[4]
    main(start_time, end_time, registration_number, ops_per_second)
