# Use only 1 GPU if available
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable progress bars

from chronos import BaseChronosPipeline, Chronos2Pipeline
from datetime import datetime
import torch
import os
from datetime import datetime
from ts_toolkit.utils.chronos_tune_input_generator import chronos_input_generator

if __name__ == "__main__":
    torch.cuda.empty_cache()
    DISTRIBUTIONS = {
        "Deterministic Trends": ['constant', 'linear', 'exponential', 'staircase'],
        "Harmonic & Periodic": ['sine', 'sawtooth', 'square', 'triangle', 'noise_patch', 'fourier'],
        "Transient Dynamics": ['damped_sine', 'growing_sine', 'chirp'],
        "Autocorrelated Processes": ['ar', 'ma', 'arma', 'sarima', 'garch', 'pink_noise', 'fractional_brownian', 'markov_chain'], 
        "I.I.D. White Noise": ['normal', 'uniform', 'poisson', 'binary', 'student_t', 'lognormal', 'laplace', 'cauchy', 'skew_normal'],
        "Stochastic Drift/Shocks": ['random_walk', 'piecewise_constant', 'gbm', 'intermittent', 'impulse', 'logistic'], 
    }

    # List of level 1 dataset types, e.g., ["univariate_lagged", "multivariate_lagged"]. If None, all types are used.
    dataset_level1 = None
    # List of level 2 dataset types, e.g., ["exponential", "noise_patch", "normal", "poisson", "random_walk"]. If None, all types are used.
    # all but Autocorrelated Processes
    dataset_level2 = DISTRIBUTIONS["Deterministic Trends"] + DISTRIBUTIONS["Harmonic & Periodic"] + DISTRIBUTIONS["Transient Dynamics"] + DISTRIBUTIONS["I.I.D. White Noise"] + DISTRIBUTIONS["Stochastic Drift/Shocks"]

    data_path = "/home/nafiseh/GitHub/timeseries-research/data/simpletime/datasets_100k"
    train_inputs, validation_inputs = chronos_input_generator(data_path, dataset_level1, dataset_level2, val_size=0.01)


    output_dir = f"results/finetuned_models/chronos2/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Load the Chronos-2 pipeline
    # GPU recommended for faster inference, but CPU is also supported using device_map="cpu"
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

    print(f"Fine-tuning Chronos-2 model. Output directory: {output_dir}")

    # Fine-tune the model
    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=48,
        num_steps=1_000_000,
        learning_rate=1e-4,
        batch_size=256,
        logging_steps=100,
        # finetune_mode="lora",
        output_dir=output_dir,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        validation_inputs=validation_inputs,
        eval_steps=100
    ) 