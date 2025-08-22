# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model Type: Supervised classification (Logistic Regression / RandomForest / AdaBoost depending on chosen final model).  
- Framework: Python, scikit-learn.  
- Pipeline: Data preprocessing (categorical encoding, label binarization) → model training → serialized with pickle → served via FastAPI.  
- Version: v1.0  

## Intended Use
- The model predicts whether an individual's annual income is `<=50K` or `>50K` based on census features.  
- Intended for educational demonstration of end-to-end ML deployment.  
- Not intended for production decisions affecting individuals (employment, credit, insurance) without careful bias/fairness validation.  

## Training Data
- Source: Census Income dataset (`census.csv`, derived from UCI Adult dataset).  
- Size: ~32,000 rows.  
- Features include: age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country.  
- Target label: Income class (`<=50K` or `>50K`).  

## Evaluation Data
- 20% hold-out split from the census dataset.  
- Stratified by income label to maintain class balance.  
- Processed using the same encoder and label binarizer fitted on training data.  

## Metrics
_Please include the metrics used and your model's performance on those metrics._  
- Metrics: Precision, Recall, F1 score (beta=1).  
- Example results on test set:  
  - Precision: ~0.74  
  - Recall: ~0.63  
  - F1 Score: ~0.68  
- Performance across slices (e.g., workclass, education, sex) was also computed; metrics vary between groups, highlighting potential bias.  

## Ethical Considerations
- The dataset encodes historical inequalities across race, gender, and education. Predictions may reinforce these biases.  
- Model should not be used in high-stakes decision-making without fairness audits.  
- Privacy concerns: although this dataset is public and de-identified, sensitive demographic attributes require careful handling in production contexts.  

## Caveats and Recommendations
- This model is for **educational purposes only**.  
- Generalization outside this dataset is limited.  
- Fairness across subgroups is inconsistent; bias mitigation is recommended for any real-world use.  
- Retraining with fresh, representative data is required before deployment in practice.  
- Users should apply interpretability methods (e.g., feature importance, SHAP values) to better understand predictions.  