# Interpretable ML Assignment

## Dataset
This project utilizes the **Heart Disease dataset** from the `imodels` library. The dataset contains medical records used to predict the presence of heart disease based on various features such as age, cholesterol levels, and maximum heart rate.

## Algorithms
The following three algorithms were selected from the `imodels` library and implemented in this project:

1. **RuleFit**  
   - Combines decision tree rules and linear models. Extracts rules from trees, converting them into interpretable if-then statements.
   - Interpretability: Provides feature importance and human-readable rules.

2. **Greedy Rule List (GRL)**  
   - Greedily adds rules one at a time based on their contribution to classification performance.
   - Interpretability: Constructs a simple list of rules, with the first matching rule being applied during inference.

3. **FIGS (Fast Interpretable Greedy-Tree Sums)**
  - FIGS is a model that constructs an ensemble of shallow decision trees in a greedy manner. Each tree is built to focus on different parts of the feature space, and their outputs are summed to make predictions. This allows FIGS to be both powerful and interpretable.
  - Interpretability: FIGS is inherently interpretable because each decision tree in the ensemble is shallow and provides clear, rule-based decisions. By limiting the depth of the trees, FIGS ensures that the rules remain understandable and interpretable by humans, making it easier to explain the reasoning behind predictions.


## Visuals
Each model is accompanied by a visual explanation that details the training and inference processes, aiding in the understanding of the algorithms. Diagrams have been included in the repository to clarify the flow of each method.

## Repository Structure
- `Assignment_4.ipynb/` : Contains the Google Colab notebook with the code for each model and documentation.
- `Images/` : Contains high-resolution visuals explaining the algorithms.
