PSYKOSE: A Motor Activity Database of Patients with Schizophrenia

The PSYKOSE dataset contains actigraphy data collected from patients with schizophrenia. In total, there are 22 participants with schizophrenia and 32 control persons. For each person in the dataset, we provide sensor data collected over several days in a row. In addition to the sensor data, we also provide some demographic data and medical assessments during the observation period. The dataset contains two directories and additional files explained below.

Dataset website: https://datasets.simula.no/psykose/

*******************
patient/ **********
*******************

The patient/ directory contains the actigraphy readings of each patient.
The first column is the timestamp, second column is the date and the third column is the activity level as recorded by the device.


*******************
control/ **********
*******************

The control/ directory contains the actigraphy readings for the control group. More information about the control group participants can be found in the DEPRESJON dataset which was part of the same study:
https://datasets.simula.no/depresjon/
https://doi.org/10.5281/zenodo.1219550

*******************
patients_info.csv *
*******************

The file patinets_info.csv contains information about the patients such as gender, age, etc. For a detailed description of each field, please, consult the paper: PSYKOSE: A Motor Activity Database of Patients with Schizophrenia.


*******************
days.csv **********
*******************

The file days.csv contains information about how many days a participant was involved in the study. The first column is the participant id and the second column the number of days. For patients, this information can also be found in patients_info.csv. For controls, the information of number of days was extracted from the DEPRESJON dataset: https://datasets.simula.no/depresjon/
https://doi.org/10.5281/zenodo.1219550

****************************
schizophrenia-features.csv *
****************************

The schizophrenia-features.csv file contains extracted features in a per day basis for each participant and the class in numeric and string format. Specifically, three features were extracted: mean, standard deviation and proportion of readings with zero values. This file was used to report classification results in the associated paper: PSYKOSE: A Motor Activity Database of Patients with Schizophrenia

For column 'class'
0: no schizophrenia
1: schizophrenia

For column 'class_str'
c0: no schizophrenia
c1: schizophrenia

====================
=====Citation=======
====================

All documents and papers that report experimental results based on the  PSYKOSE Dataset, should include a reference to this paper:

P. Jakobsen, E. Garcia-Ceja, L.A. Stabell K.J. Oedegaard, J.O. Berle, S.A. Hicks, V. Thambawita, P. Halvorsen, O.B. Fasmer, M. Riegler., 
"PSYKOSE: A Motor Activity Database of Patients with Schizophrenia", 2020.


BibTex entry:
@misc{Jakobsen2020, title={Psykose: A Motor Activity Database of Patients with Schizophrenia}, url={}, DOI={}, publisher={OSF Preprints}, author={Jakobsen, Petter and Garcia-Ceja, Enrique and Stabell, Lena Antonsen and Oedegaard, Ketil Joachim and Berle, Jan Oystein and Hicks, Steven and Thambawita, Vajira and Halvorsen, P{\aa}l and Fasmer, Ole Bernt and Riegler, Michael Alexander}, year={2020}, month={Feb} }