import yaml
import matplotlib.pyplot as plt

def plot_results(results: dict) -> None:
    model_names = []
    avg_inference_times = []
    std_dev_inference_times = []

    for model_name, result in results.items():
        model_names.append(model_name.replace('../models/', ''))
        avg_inference_times.append(result['average_inference_time'] * 1000) 
        std_dev_inference_times.append(result['standard_deviation_inference_time'] * 1000)
        
    fig, ax = plt.subplots()
    bar_width = 0.5
    bars = ax.bar(model_names, avg_inference_times, yerr=std_dev_inference_times, align='center', alpha=0.5, ecolor='red', capsize=bar_width*30, width=bar_width, color='lightgrey')
    for bar in bars:
        bar.set_edgecolor('black')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_xticks(range(len(model_names)))
    ax.set_title('Inference Time Comparison')
    ax.grid(alpha=0.2)
    ax.axhline(y=100, color='green', alpha=0.2) 
    plt.tight_layout()
    plt.show()

with open('results.yaml', 'r') as f:
    results = yaml.safe_load(f)

plot_results(results)