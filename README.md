# Meal Planner using Agentic RAG

This project is a meal planning application that leverages advanced AI models and tools to create customized meal plans based on user-provided ingredients, dietary restrictions, allergy information, and daily protein targets. The application uses Qdrant for vector database management, Qwen2.5 Coder 32B for language processing, and Mem0 for memory management.

## Features

- **Customized Meal Plans**: Generate meal plans based on user inputs.
- **Ingredient Utilization**: Maximize the use of user-provided ingredients.
- **Dietary Compliance**: Ensure recipes adhere to dietary restrictions and allergies.
- **Nutritional Information**: Provide detailed nutritional information for each meal.
- **Memory Management**: Store and retrieve chat history using Mem0.

## Project Structure

- `dataset_preprocessing.ipynb`: Jupyter notebook for preprocessing the dataset.
- `upload_2_qdrant.ipynb`: Jupyter notebook for uploading data to Qdrant.
- `agentic_rag.py`: Python script for initializing and managing agents.
- `app.py`: Streamlit application for user interaction.

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/pranav-0309/meal-planner-fyp.git
    cd meal_planner
    ```

2.  **Download the dataset**:
    - Go to the [download page](https://recipenlg.cs.put.poznan.pl/dataset).
    - Accept Terms and Conditions and download zip file.
    - Unpack the zip file and you'll get the `full_dataset.csv` file in the dataset directory. 

3. **Create your virtual environment**:
    You can create a new virtual environment anyway you like but I made it using the following code (You need to have Anaconda installed):
    ```bash
    conda create -n venv python=3.12 -y (to create the venv)
    conda activate venv (to activate it)
    conda deactivate (to deactivate it)
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Make all the neccessary accounts to get all the API keys and create a `.env` file and add the necessary API keys and configurations. And make sure to import the API keys and configurations in the code where ever necessary.

4. **Run the application**:
    - First run all the cells in the `dataset_preprocessing.ipynb` file. This will allow you to get the preprocessed dataset and the dataset with 1,000,000 records.
    - Next, run all the cells in the `upload_2_qdrant.ipynb` file so that your vector database is ready.
    - Then copy and paste the below code into your terminal to run the app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Enter User ID**: Start by entering your user ID.
2. **Input Ingredients and Preferences**: Provide details about your ingredients, dietary restrictions, allergies, and daily protein target.
3. **Generate Meal Plan**: The application will generate a meal plan based on your inputs.
4. **Clear The Chat:** A button labelled "Clear" is located next to the user ID text box. You can use this to clear the chat at anytime and start the conversation again.

## Dependencies

- `streamlit`
- `sentence_transformers`
- `qdrant_client`
- `torch`
- `smolagents`
- `dotenv`
- `mem0`

## License

This project is licensed under a custom license. You may use the code for personal, non-commercial purposes only (e.g., running the app locally). Commercial use and redistribution are strictly prohibited. See the [LICENSE](./LICENSE) file for more details.

## Disclaimer

- This project was built as a proof of concept/demo of Pranav Krishnakumar's research on Agentic RAG for his final year project. It is meant only to demonstrate that you can use AI Agents to perform RAG in a much more accurate manner compared to traditional RAG.
- At times the AI's responses could be inaccurate so please cross check any import facts.
- While it's rare, the AI may occasionally produce responses that lack proper formatting. Please keep in mind that the AI is instructed to provide formatted answers, so this is likely not due to an error in the code itself.
