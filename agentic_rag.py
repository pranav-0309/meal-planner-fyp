# All the imports for this file
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import torch
from smolagents import HfApiModel, Tool, ToolCallingAgent, ManagedAgent, CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
from smolagents.prompts import CODE_SYSTEM_PROMPT

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the embedding model on GPU
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_model.to(device)

# Initialize the Qdrant client
qdrant = QdrantClient(
    url="create an account in https://qdrant.tech/ and get your own cluster's url",  
    api_key="get your own api key from the same account for the same cluster"  
)

# Initialize the LLM
llm = HfApiModel(
    model_id = "Qwen/Qwen2.5-Coder-32B-Instruct",
    token = "go to https://huggingface.co/settings/tokens login to your account and get an api key"
    )

# Initializing the web searching agent to retrieve the nutritional information of the food items
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model = llm
    )

# Initialize the Managed Agent for web searching
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search_agent",
    description="This Agent can be used to search the web for nutritional information of any food item. In a worst case senario, you can also use it to search anything that could aid you in creating the meal plan."
)

# Creating the retriever tool class to retrieve the documents from the knowledge base
class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question."
        }
    }
    output_type = "string"

    def __init__(self, vectordb: QdrantClient, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        # Encode the query to vector
        query_vector = embedding_model.encode(query).tolist() # type: ignore

        data = self.vectordb.search(
            collection_name="recipe_data",
            query_vector=query_vector,
            limit=5,
        )

        return f"Retrieved documents:\n{data}"

# Initializing the retriever tool
qdrant_retriever_tool = RetrieverTool(vectordb=qdrant)

# Initializing the retriever agent
retriever_agent = ToolCallingAgent(
    tools=[qdrant_retriever_tool], model=llm
)

# Initializing the Managed Agent for the retriever agent
managed_retriever_agent = ManagedAgent(
    agent=retriever_agent,
    name="retriever_agent",
    description="""Retrieves documents from the knowledge base for you that are close to the input query.
                    Give it your query as an argument. The knowledge base includes recipes with their title,
                    ingredients, and directions (steps to follow).
                """,
)

# Setting the prompt for the meal planning agent
custom_prompt = """
Act like a professional nutritionist and meal prep specialist with expertise in creating efficient and customized weekly meal plans. 

Task: 
The user will provide details about the ingredients they have, dietary restrictions, allergy information, and their daily protein target. 
Using the vector database provided to you, create three distinct recipes—one for breakfast, one for lunch, and one for dinner—that the user will repeat daily for 7 days. 
Each recipe should include specific ingredient quantities for meal prepping for 7 servings, ensuring each serving meets or closely aligns with the user's daily protein target when divided across meals.

Instructions:
Step 1: Analyze the user's input to understand the ingredients they have, their dietary restrictions, allergies, and daily protein target.
Step 2: Query the vector database appropriately to retrieve relevant recipes based on the user's input.
Step 3: Follow the guidelines stated below and create the meal plan for the user.

Guidelines: 
1. Dietary and Allergy Compliance: Ensure recipes strictly adhere to the user's dietary restrictions and allergies.  
2. Protein Target: Use the daily protein target provided by the user to guide recipe creation. Ensure the protein content of each serving from breakfast, lunch, and dinner combines to meet or closely approximate the target. Distribute the protein evenly across meals or based on meal size.  
3. Ingredient Utilization: Maximize the use of the user-provided ingredients. If additional ingredients are required, include them in the ingredients list and tag them as "Must be purchased."  
4. Nutritional Information: Provide nutritional details (calories, protein, fats, and carbohydrates) for one serving of each meal.  
5. Meal Prep Quantities: Clearly indicate the total quantity of each ingredient needed to prepare all 7 servings of the recipe.  
6. Preparation Instructions: Include concise, step-by-step instructions for preparing each recipe, suitable for batch cooking and storage.
7. Formatting: Present the meal plan in a clear, organized and human readable format.
8. Topic Relevance: Respond only to inquiries directly related to the user's meal plan or food-related questions. Avoid addressing any off-topic questions and inform the user you cannot answer those type of questions.
"""

# Combining the original system prompt with my own prompt:
final_prompt = CODE_SYSTEM_PROMPT + custom_prompt

# Initializing the CodeAgent for the meal planner
meal_planner_agent = CodeAgent(
    tools=[],
    model=llm,
    system_prompt=final_prompt,
    managed_agents=[managed_web_agent, managed_retriever_agent]
)