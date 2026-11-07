# Connect Four AI

An AI that taught itself Connect Four by playing against itself, and a small web app where you can play against it.

**[Play it here](https://try-to-connect-four.streamlit.app)**

## What's inside

- **The notebook** (`connect_four_deep_rl.ipynb`) trains the AI with deep reinforcement learning. It starts knowing only the rules and improves over thousands of games against itself. It also compares different ways of learning and tests the AI against other bots.
- **A stronger version** with a bigger network and a few moves of look-ahead. It beats a standard search bot (MCTS with 200 simulations) about 9 out of 10 times, against roughly 1 in 10 for the first version.
- **The web app** (`app.py`) lets you play against it in the browser. Under **Settings** you can choose who moves first, how far the AI looks ahead, and show how it rates every column.

## Training it yourself

The trained models are in `models/`. To train new ones, open the notebook in Google Colab with a GPU runtime and run all cells. It takes about an hour, and the last section downloads the new models.
