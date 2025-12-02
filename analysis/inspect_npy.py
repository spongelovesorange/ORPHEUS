import numpy as np

try:
    print("\n--- Checking any_conf_final_preds.npy ---")
    data = np.load('/data/Matcha-main/inference_results/run_1stp_experiment/any_conf_final_preds.npy', allow_pickle=True)[0]
    print(f"Keys in data: {list(data.keys())}")
    
    if len(data) > 0:
        first_key = list(data.keys())[0]
        print(f"First key: {first_key}")
        sample_data = data[first_key]
        print(f"Keys in sample_data: {list(sample_data.keys())}")
        
        if 'sample_metrics' in sample_data:
            samples = sample_data['sample_metrics']
            print(f"Number of samples: {len(samples)}")
            if len(samples) > 0:
                print(f"Keys in first sample: {list(samples[0].keys())}")
                if 'flow_field_05' in samples[0]:
                    flow = samples[0]['flow_field_05']
                    print(f"Flow field shape: {flow.shape}")
                    print(f"Mean flow magnitude: {np.linalg.norm(flow, axis=1).mean():.4f}")
                else:
                    print("Flow field NOT found in first sample")
        else:
            print("sample_metrics NOT found in sample_data")
            
except Exception as e:
    print(f"Error: {e}")

