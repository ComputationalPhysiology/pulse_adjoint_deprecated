#Abel scripts#
This folder contains script for running the code on the Abel cluster.

The main script is run_mulitple.py where you define the parameters sets you want to test.
If you want to run multiple fiber angles you can run the script save_patient_data.py
first in serial and then run the script run_mulitple in parallell. The reason for this 
is because fiber generation only run in serial. 