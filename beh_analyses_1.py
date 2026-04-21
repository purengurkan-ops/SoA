#-------------------------
# This file includes pseudocode / guidelines for the behavioral analyses of the control detection & memory project
# --------------------

# The packages you need 
import os #the package that allows python to read files from local folders
import numpy as np #the package that allows you to perform numerical operations
import pandas as pd #the package that allows you to work with dataframes
import matplotlib.pyplot as plt #the package that allows you to create plots
import seaborn as sns #the package that allows you to create more advanced plots
from pathlib import Path

#-------------------------
# Import data
#-------------------------

DATA_DIR = Path(r"pilot_data")

# same goes for the output directory. I have just created a new empty folder called analysis_output in our repo. set that as OUTPUT_DIR below

# insert code here

# Participant filter: set to a non-empty list to restrict analyses to specific
# participant IDs, e.g. PARTICIPANT_FILTER = [2, 3].
# Leave as an empty list [] to include ALL participants found in data files.
PARTICIPANT_FILTER = []

#-------------------------
# Load data
#-------------------------

# 1. Find all data files matching the specific naming pattern. 
# DATA_DIR.glob tells Python to look inside DATA_DIR and fetch files ending with .csv
# CDmem_1_* is the naming pattern for data coming from the control detection task NOT recognition. 
all_files = list(DATA_DIR.glob("CDmem_1_*.csv"))

# 2. Create an empty list to store the dataframes for each participant
# a list is the same structure as the participant filter above :)

# insert code here

# 3. Loop through each file path that we found
# a for loop does whatever is indented below it, for each item in the list
# so this one will do the indented actions for everything in all_files
for file_path in all_files:
    
    # Read the CSV file into a pandas DataFrame
    # a pandas dataframe is basically a table with rows and columns
    # you need to give it a name, usually df
    df = pd.read_csv(file_path)
    
    # Add this DataFrame to the empty list you created.
    # you do this via writing the name of the empty list and then .append(df), single line of code
    # .append() is a "method" that adds the item in parantheses to the end of the list
    
    # insert code here


# up to this point, we have loaded all the data in the data folder.
# 4.  now we need to filter participants if you set numbers in PARTICIPANT_FILTER list above

if PARTICIPANT_FILTER:
    # Use list comprehension to keep only data for included participants
    # change "thelistyoucreated" to whatever name you gave your list above
    thelistyoucreated = [df for df in thelistyoucreated if df["participant"].iloc[0] in PARTICIPANT_FILTER]


# 5. recall that we have everything in a list. but for stats, we need everything in a dataframe (i.e., rows, columns, etc. so we can use column names etc later)
# we do this with pd.concat()
# pd.concat() is a function that takes a list of dataframes and concatenates them into a single dataframe
# ignore_index = True means that we want to create a new index for the combined dataframe

# change "thelistyoucreated" to whatever name you gave your list above
if thelistyoucreated: # This checks if the list is not empty
    data = pd.concat(thelistyoucreated, ignore_index=True)
    print(f"Successfully loaded {len(thelistyoucreated)} data files! Total rows: {len(data)}")
    # len() gives the length of a list. len(thelistyoucreated) gives the number of lists in the listyoucreated. recall that thelistyoucreated was created to store all data from the folder.
    # similarly, len(data) gives the number of rows in the dataframe.
else:
    print("Warning: No data files found. Please check your DATA_DIR folder.") 

# ============================================================================
# EXCLUSION CRITERION 1 - TIMEOUT RATE
# ============================================================================
# We want to exclude participants who have 50% or more timeout trials 
# in either the 'low' or 'high' control condition during the 'test' phase.
# set a threshold for the timeout rate. basically a variable that equals 0.50

# insert code here

# 1. Filter the data to only include the "test" phase 
# the .copy one creates a copy so that the original doesnt get modified
test_data = data[data["phase"] == "test"].copy()

# 2. Make sure the "is_timeout" column contains only True or False
# Sometimes it loads as text like "True" or "False", so we force it to be a boolean
# so it does not take it as the word "true" but the logical true
test_data["is_timeout"] = test_data["is_timeout"].astype(str).str.strip().str.lower() == "true"

# 3. Calculate the timeout rate per participant per condition. 
# We can do this by taking the mean of the "is_timeout" column (since True is 1 and False is 0)

# timeout_rate = ........
# insert more code here

# 4. Find the rows where the timeout rate is greater than or equal to our threshold

# insert code here

# 5. Get a simple list of their unique participant IDs


print("\nEXCLUSION CRITERION 1: Timeout Rate")
# change "listofexcludedparticipants" to whatever you named that list at step 5
if len(listofexcludedparticipants) > 0:
    print(f"  -> Excluded {len(listofexcludedparticipants)} participants: {listofexcludedparticipants}")
else:
    print("  -> No participants were excluded.")

# 6. Remove these participants from our main 'data' dataframe
# similar to how we filtered stuff in the first step, but this time we use the '~' symbol
# The '~' symbol means "NOT" (keep data where participant is NOT in the list)
# don't forget to use .copy() again

# insert code here

#-----------------------------------
# NOW TRY TO LOAD AND FILTER THE RECOGNITION DATA
# file naming convention is: CDmem_*_recognition.csv
# recall that the exclusion criteria for recognition data is trial based:
# following Ren et al., 2026, any trial that has a RT of +- 3SD of the **participant mean** should be excluded

# insert code here









# NEXT STEP WILL BE TO MATCH EXCLUSIONS FROM BOTH DATASETS AND RUN ANALYSES