# **Disease Classification using Random Forest Classification and Dense Nuero Networks**

Gerardo Sillas<br>
Machine Learning<br>
5/13/2025<br>

COLAB VIEWING LINK: https://colab.research.google.com/drive/1fR75OATVe4qDD84kCyL7kIJajlwF27m_?usp=sharing

# **Premise:**
My project classifies disease based on the symptoms that are provided. The dataset used in this project is the *Mendeley Disease and Symptoms* dataset with a total of 773 unique diseases and 377 symptoms, with around 247,000 rows. The data in this dataset are 1, 0, representations of whether the example disease has that specific symptom (1) or not (0). This will be implemented using a Random Forest Classifier model and a Dense Neural Network model. Both models were chosen because I belive they offer a unique approach that is tailored to this data set and I was also curious to see which would perform better.

# **Problem:**
The problem my project is aimed at solving
- **Incorrect diagnosis** - Medical professionals occasionally provide inaccurate diagnoses. By cross-referencing the model's predictions with the doctor's assessment, the frequency of these mistakes can be reduced. This also indirectly helps to tackle related challenges such as repeated doctor visits, unnecessary medical treatments, excessive healthcare costs, and doctor burnout.
- **Time efficient** - This project enhances the speed of the diagnosis process, enabling doctors to treat more patients in less time. With a more efficient workflow, doctors may experience reduced work hours, which can help mitigate burnout and improve work-life balance.

 # **Hypothesis**:
I hypothesize that the Dense Neural Network will perform better than Random Forest Classifier because Dense Neural Networks are at capturing complex patterns. I've also observed that in previous exercises Dense Neural Networks out performed Random Forest Classifier.

# **What was done:**
**Data Processing**:
  Downloaded and preprocessed the dataset, creating the input features (x), target labels (y), and the one-hot representation for the target labels (y_test_ohr and y_train_ohr).

**Model Development:**
- **Random Forest Classifier:** Implemented a Random Forest Classification Model and trained it using the processed data. Fune-tuned the model's parameters to maximize accuracy
- **Dense Neural Network:** Developed a Dense Neural Network model and trained it on the dataset. Optimized the model's layer architecture and adjust the parameters to maximize performance.

**Model Evaluation:**
Compared the performance (accuracy) of both the Random Forest and Dense Neural Network models to determine which one provided best results for classifying the diseases based on their symptoms.

# **Results:**
**Random Forest Classification:** This model exhibits

Training accuracy: 0.9120

Test accuracy: 0.8408

------------------------------------------------------------------------------
**Dense Neural Networks:** This model exhibits,


Training Accuracy: 0.8584

Test Accuracy: 0.8621

Training Loss: 0.3598

Test Loss: 0.3275

------------------------------------------------------------------------------

Highest Training Accuracy: 0.8589

Highest Test Accuracy: 0.8622

Lowest Training Loss: 0.3598

Lowest Test Loss: 0.3250

------------------------------------------------------------------------------
**Final Thoughts:**

Based on the results, the Dense Neural Network (DNN) performed slightly better on this dataset, achieving a test accuracy of 86.21%, which supports my hypothesis. I believe the DNN outperformed the Random Forest Classifier because it was able to learn complex patterns formed by the different combinations of symptoms associated with each disease. While some diseases share symptoms, others have more distinct ones, and the DNN is better at capturing these subtle differences. However, the Random Forest still performed competitively, likely because decision trees handle binary data well, making them a good fit for this dataset.

# **What was learned:**
- With this project I was able to gain further experience and understanding of Random Forest Classifiers and DNN.
- I learned a lot about the different parameters provided by DNN and Random Forest Classifier. By tailoring the parameters for both models I got real time feedback on how each parameter affects the performance of the model.
- Also became familiar with coding metrics, and got to see how they provide deeper insight into model performance.
- Although not the main focus of the project, I also learned a lot about file reading, which was very prominent when working with the dataset.

# **Resources**
- Stark, Bran. “Disease and Symptoms Dataset 2023.” Mendeley Data, Mendeley Data, 3 Mar. 2025, data.mendeley.com/datasets/2cxccsxydc/1.
- Fuentes, Olac. "Dense FFNN for CIFAR10." Google Colab, Google, 2025, colab.research.google.com/drive/1gopXL1sRMZm35M8hhjq8eMioOTQRRGd3#scrollTo=6vCoA01pka10.
- Fuentes, Olac. “Random Forest.” Google Colab, Google, 2025, colab.research.google.com/drive/1Ih6vq3yqYfj5fvhdLcVGH15UdBlDf13h#scrollTo=1pOlN3lI13dJ.
-“Matplotlib.Pyplot#.” Matplotlib.Pyplot - Matplotlib 3.5.3 Documentation, matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html. Accessed 16 May 2025.
- “HTTP for HumansTM¶.” Requests, requests.readthedocs.io/en/latest/. Accessed 16 May 2025.
-“CSV - Csv File Reading and Writing.” Python Documentation, docs.python.org/3/library/csv.html. Accessed 16 May 2025.
- Google Colab, Google, colab.research.google.com/notebooks/io.ipynb. Accessed 16 May 2025.
