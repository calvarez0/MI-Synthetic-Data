# MI Synthetic Data

"main.py" is the code for running the automatic MI pair generation.

"cannabis_generation.log" has the terminal output of the whole process.

"mi_dialogue_archive.json" has information about the pairs and their scores.

"mi_training_data.json" has all the pairs ordered in decreasing rank.

"first_alpaca" is folder that holds the correct format for fine-tuning the LLM (I think the fine-tuning package is called alpaca).

"convert_to_alpaca.py" is a python script that converts the top 100 pairs from mi_training_data.json to mi_alpaca_format.json.