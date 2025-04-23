This pakage includes all codes to generate results in the following paper:

Reece A. McDorman, Bhoj Raj Thapa, Jenna Kim, and Jihye Bae. "Transfer Learning in EEG-based Reinforcement Learning Brain Machine Interfaces via Q-learning Kernel Temporal Differences." 2025 Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 2025.

If you use any of the codes published in this repository, partially or fully, you MUST cite the above conference paper.

====================================
README
====================================
This README.txt provides an overview of the files contained in this repository, instruction on how to use them effectively, 
and specific details on how to set up and run the code to generate the results as described in the paper.

------------------------------------
ORGANIZATION
------------------------------------

All codes used to generate the results of the paper are in this folder. 
Before running them, variables at the top of each code need to be changed according to the location of the datafiles and preferred location of results.

1. **Code_Step01_EEG_FREEFORM.m**
   - **dataPath:** Needs to be set to the full path of the folder 
     containing the FREEFORM data files.
   - **resultPath:** Needs to be set to a folder of your choice where 
     the results of feature generation, training, testing, and plotting 
     will be stored. 
2. **Code_Step02_KTD_EEG_wQuantization_TLsource.m**
   - **featurePath:** Should be set to the `resultPath` defined in 
     `Code_Step01_EEG_FREEFORM.m`.
3. **Code_Step03_KTD_EEG_wQuantization_TLtraining.m**
   - **featurePath:** Should also be set to the `resultPath` defined 
     in `Code_Step01_EEG_FREEFORM.m`.
4. **Code_Step04_Results_Plotting.m**
   - **parentTLResultPath:** Should be set to the `resultPath` defined 
     in the previous steps.

Once these variables are correctly set, the code is ready to be tested.

------------------------------------
RUNNING THE CODES
------------------------------------
1. **Feature Generation (Code_Step01_EEG_FREEFORM.m):**
   - This script generates FTA and RAW features from the EEG data in a 
     specified file and for a specified time window in each trial.
   - **Variables:**
     - `dataName`: Specifies the data file from which to extract EEG 
       features.
     - `tstart` and `tend`: Define the start and end of the time 
       extracted for each trial relative to t = 0 (movement onset). 
     - `featureName`: Define a type of feature you want to generate (FTA or RAW)
   - **Usage:** Run this script first to preprocess your EEG data and 
     generate the necessary features. Total combinations could be as follows:
     Combination) dataName/featureName/[tstart,tend]
     1) FREEFORMSubjectB1511112StLRHand/RAW/[-0.85,0]
     2) FREEFORMSubjectB1511112StLRHand/RAW/[0,0.85]
     3) FREEFORMSubjectB1511112StLRHand/FTA/[-0.85,0]
     4) FREEFORMSubjectB1511112StLRHand/FTA/[0,0.85]
     5) FREEFORMSubjectC1512082StLRHand/RAW/[-0.85,0]
     6) FREEFORMSubjectC1512082StLRHand/RAW/[0,0.85]
     7) FREEFORMSubjectC1512082StLRHand/FTA/[-0.85,0]
     8) FREEFORMSubjectC1512082StLRHand/FTA/[0,0.85]
     9) FREEFORMSubjectC1512102StLRHand/RAW/[-0.85,0]
     10) FREEFORMSubjectC1512102StLRHand/RAW/[0,0.85]
     11) FREEFORMSubjectC1512102StLRHand/FTA/[-0.85,0]
     12) FREEFORMSubjectC1512102StLRHand/FTA/[0,0.85]

2. **Source Domain Training (Code_Step02_KTD_EEG_wQuantization_TLsource.m):**
   - This script trains Q-KTD using the generated features and stores 
     weights and units to be used for transfer learning in the target 
     domain.
   - **Variables:**
     - `dataName`: Specifies the name of the data file corresponding to 
       the session from which you want to train Q-KTD (source dataset).
     - `featureName`: Specifies the feature (FTA or RAW).
     - `tstart` and `tend`: Specify the time window of the training 
       features to be used.
   - **Usage:** Run this script after feature generation to train the 
     source model. Total combinations could be as follows:
     Combination) dataName/featureName/[tstart,tend]
     1) FREEFORMSubjectB1511112StLRHand/RAW/[-0.85,0]
     2) FREEFORMSubjectB1511112StLRHand/RAW/[0,0.85]
     3) FREEFORMSubjectB1511112StLRHand/FTA/[-0.85,0]
     4) FREEFORMSubjectB1511112StLRHand/FTA/[0,0.85]
     5) FREEFORMSubjectC1512082StLRHand/RAW/[-0.85,0]
     6) FREEFORMSubjectC1512082StLRHand/RAW/[0,0.85]
     7) FREEFORMSubjectC1512082StLRHand/FTA/[-0.85,0]
     8) FREEFORMSubjectC1512082StLRHand/FTA/[0,0.85]
     9) FREEFORMSubjectC1512102StLRHand/RAW/[-0.85,0]
     10) FREEFORMSubjectC1512102StLRHand/RAW/[0,0.85]
     11) FREEFORMSubjectC1512102StLRHand/FTA/[-0.85,0]
     12) FREEFORMSubjectC1512102StLRHand/FTA/[0,0.85]

3. **Target Domain Training (Code_Step03_KTD_EEG_wQuantization_TLtraining.m):**
   - This script uses weights and units generated in the previous step 
     to initialize and train Q-KTD on a new dataset (target domain).
   - **Variables:**
     - `dataNameSource` and `dataNameTraining`: Define the source and 
       target datasets, respectively.
     - `featureName`, `tstartSource`, and `tendSource`: These variables 
       are defined similarly to the previous steps and apply to both 
       the source and target datasets.
     - `tstart` and `tend`: Specify the time window of the training 
       features to be used.
   - **Note:** To plot learning curves for comparison with and without 
     TL, both the source and target datasets, along with specified 
     combinations of windows and features, need to be run in 
     `Code_Step02_KTD_EEG_wQuantization_TLsource.m` first. It is 
     recommended to run all dataset and feature/window combinations 
     (2 features * 2 windows * 3 subjects = 12 total cases) in the 
     source step before proceeding with TL experiments.
   - **Usage:** Execute this script to train the TL model on the target 
     dataset using the source-initialized model. Total combinations could be 
     as follows:
     Combination) dataNameSource/dataNameTraining/featureName/[tstart,tend]
     1) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512082StLRHand/RAW/[-0.85,0]
     2) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512102StLRHand/RAW/[-0.85,0]
     3) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512082StLRHand/RAW/[0,0.85]
     4) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512102StLRHand/RAW/[0,0.85]
     5) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512082StLRHand/FTA/[-0.85,0]
     6) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512102StLRHand/FTA/[-0.85,0]
     7) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512082StLRHand/FTA/[0,0.85]
     8) FREEFORMSubjectB1511112StLRHand/FREEFORMSubjectC1512102StLRHand/FTA/[0,0.85]
     9) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectB1511112StLRHand/RAW/[-0.85,0]
     10) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectC1512102StLRHand/RAW/[-0.85,0]
     11) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectB1511112StLRHand/RAW/[0,0.85]
     12) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectC1512102StLRHand/RAW/[0,0.85]
     13) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectB1511112StLRHand/FTA/[-0.85,0]
     14) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectC1512102StLRHand/FTA/[-0.85,0]
     15) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectB1511112StLRHand/FTA/[0,0.85]
     16) FREEFORMSubjectC1512082StLRHand/FREEFORMSubjectC1512102StLRHand/FTA/[0,0.85]
     17) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectB1511112StLRHand/RAW/[-0.85,0]
     18) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectC1512082StLRHand/RAW/[-0.85,0]
     19) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectB1511112StLRHand/RAW/[0,0.85]
     20) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectC1512082StLRHand/RAW/[0,0.85]
     21) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectB1511112StLRHand/FTA/[-0.85,0]
     22) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectC1512082StLRHand/FTA/[-0.85,0]
     23) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectB1511112StLRHand/FTA/[0,0.85]
     24) FREEFORMSubjectC1512102StLRHand/FREEFORMSubjectC1512082StLRHand/FTA/[0,0.85]

4. **Results Plotting (Code_Step04_Results_Plotting.m):**
   - This script generates a bar plot to compare results with and 
     without TL for all cases, and saves it in `resultPath -> 
     graphTLResults`.
   - **Usage:** This is the final step of the experiment. Run this 
     script to visualize the results of your analysis. Note that it requires
     all the results from previous steps.

------------------------------------------------------------------
ADDITIONAL INFORMATION
------------------------------------------------------------------

For further clarifications, troubleshooting, or additional information, 
please refer to the corresponding author. 
**Jihye Bae**  
Email: jihye.bae@uky.edu
==================================================================
