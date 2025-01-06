# WisconsinDiagnosticsBreastCancer

 Summary of Findings
This project focused on developing machine learning models to classify breast tumors as benign or malignant using the Wisconsin Breast Cancer dataset, which includes 569 samples with 30 numerical features describing tumor characteristics.

Models and Performance Metrics
Logistic Regression

Accuracy: 98.82%
Sensitivity (Benign): 98.36%
Specificity (Malignant): 99.08%
Kappa: 0.9744
Logistic Regression emerged as the best-performing model, excelling in all metrics.
Support Vector Machine (SVM)

Accuracy: 97.06%
Sensitivity: 97.2%
Specificity: 96.83%
Kappa: 0.9372
SVM performed robustly, with high sensitivity and specificity, but slightly lower accuracy compared to Logistic Regression.
Random Forest

Accuracy: 96.47%
Sensitivity: 97.2%
Specificity: 95.24%
Kappa: 0.92
Random Forest balanced performance well, benefiting from ensemble learning.
Decision Tree

Accuracy: 91.76%
Sensitivity: 93.46%
Specificity: 88.89%
Kappa: 0.8235
While interpretable, the Decision Tree underperformed compared to ensemble and advanced models.
k-Nearest Neighbor (KNN)

Accuracy: 91.18%
Sensitivity: 92.72%
Specificity: 88.89%
Kappa: 0.811
KNN achieved moderate accuracy but lagged behind in sensitivity and specificity.
Key Observations
Best Model: Logistic Regression demonstrated the highest accuracy and agreement (Kappa) with actual classifications, making it the most reliable for this task.
SVM: Achieved competitive accuracy with excellent sensitivity and specificity, suitable for tasks requiring robust classification.
Hyperparameter Tuning: Optimization of parameters improved Random Forest and SVM but had limited impact on overall rankings.
Conclusion
Logistic Regression is the most effective model for classifying tumors in this dataset, combining high accuracy (98.82%) with interpretability. SVM also offers strong performance and could be preferred in cases where non-linear boundaries are critical. Future work could explore advanced ensemble techniques or hybrid models to further enhance classification performance.
