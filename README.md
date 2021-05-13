# README - BIG-LID
Implementation of the poisoning attack againts nonlinear regression and the BIG-LID defense.
- **datasets**: Contains three datasets: Appliances, House and Heart Disease
- **Main scripts**:
        -  attack_script.py - contains the attack algorithm
        -  big_lid.py - contains the defense algorithm
        -  compare_defenses.py - compares big-lid against other defense algorithms
-  **Support scripts**:
        -  defenses.py - implementation of defenses
        -  attack_utilities.py - functions used by the attack algorithm
        -  lid_utils.py - utility functions for calculating LID
        -  utils.py - general utility functions
        
Start by running the attack script on a dataset.
Then execute big_lid.py or compare_defenses.py. The required packages are listed in requirements.txt.