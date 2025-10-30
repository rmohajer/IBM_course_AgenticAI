#%%
import os
import yaml
import base64
from crewai import Agent, Crew, Process, Task
from crewai import LLM
from crewai.project import CrewBase, agent, crew, task
from src.tools import (
    ExtractIngredientsTool, 
    FilterIngredientsTool, 
    DietaryFilterTool,
    NutrientAnalysisTool
)

from src.models import RecipeSuggestionOutput, NutrientAnalysisOutput 
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import sys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model_id = "llama-3.1-8b-instant"


model = ChatGroq(model=model_id,
            temperature=0,
            max_tokens=2000,
            timeout=None,
            max_retries=2,
            verbose=1)

# llm = LLM(model="groq/llama-3.1-8b-instant")


# Get the absolute path to the config directory
CONFIG_DIR = os.path.join( "src/config")
# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up to project root (if this script is inside src/)
PROJECT_ROOT = os.path.dirname(BASE_DIR)



sys.path.append(os.path.join(PROJECT_ROOT, "src"))




#%%

@CrewBase
class BaseNourishBotCrew:
    # agents_config_path = os.path.join("src/config", 'agents.yaml')
    # tasks_config_path = os.path.join("src/config", 'tasks.yaml')
    # Build config paths safely
    agents_config_path = os.path.join(PROJECT_ROOT, "src", "config", "agents.yaml")
    tasks_config_path = os.path.join(PROJECT_ROOT, "src", "config", "tasks.yaml")
    print("Agents path:", agents_config_path)
    print("Tasks path:", tasks_config_path)
    def __init__(self, image_data, dietary_restrictions: str = None):
        self.image_data = image_data
        self.dietary_restrictions = dietary_restrictions

        with open(self.agents_config_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)
        
        with open(self.tasks_config_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)
            
    @agent
    def ingredient_detection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['ingredient_detection_agent'],
            tools=[
                ExtractIngredientsTool.extract_ingredient, 
                FilterIngredientsTool.filter_ingredients
            ],
            allow_delegation=False,
            verbose=True
        )
    
    @agent
    def dietary_filtering_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['dietary_filtering_agent'],
            tools=[DietaryFilterTool.filter_based_on_restrictions],
            allow_delegation=True,
            max_iter=6,
            verbose=True
        )
    
    @agent
    def nutrient_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['nutrient_analysis_agent'],
            tools=[NutrientAnalysisTool.analyze_image],
            allow_delegation=False,
            max_iter=4,
            verbose=True
        )
    
    @agent
    def recipe_suggestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['recipe_suggestion_agent'],
            allow_delegation=False,
            verbose=True
        )
    
    @task
    def ingredient_detection_task(self) -> Task:
        task_config = self.tasks_config['ingredient_detection_task']

        return Task(
            description=task_config['description'],
            agent=self.ingredient_detection_agent(),
            expected_output=task_config['expected_output']
        )
    
    @task
    def dietary_filtering_task(self) -> Task:
        task_config = self.tasks_config['dietary_filtering_task']

        return Task(
            description=task_config['description'],
            agent=self.dietary_filtering_agent(),
            depends_on=['ingredient_detection_task'],
            input_data=lambda outputs: {
                'ingredients': outputs['ingredient_detection_task'],
                'dietary_restrictions': self.dietary_restrictions
            },
            expected_output=task_config['expected_output']
        )
    
    @task
    def nutrient_analysis_task(self) -> Task:
        task_config = self.tasks_config['nutrient_analysis_task']

        return Task(
            description=task_config['description'],
            agent=self.nutrient_analysis_agent(),
            expected_output=task_config['expected_output'],
            output_json=NutrientAnalysisOutput
        )
    
    @task
    def recipe_suggestion_task(self) -> Task:
        task_config = self.tasks_config['recipe_suggestion_task']

        return Task(
            description=task_config['description'],
            agent=self.recipe_suggestion_agent(),
            depends_on=['dietary_filtering_task'],
            input_data=lambda outputs: {
                'filtered_ingredients': outputs['dietary_filtering_task']
            },
            expected_output=task_config['expected_output'],
            output_json=RecipeSuggestionOutput
        )


@CrewBase
class NourishBotRecipeCrew(BaseNourishBotCrew):

    @crew
    def crew(self) -> Crew:
        tasks = [
            self.ingredient_detection_task(),
            self.dietary_filtering_task(),
            self.recipe_suggestion_task()
        ]

        agents = [
            self.ingredient_detection_agent(),
            self.dietary_filtering_agent(),
            self.recipe_suggestion_agent()
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )


@CrewBase
class NourishBotAnalysisCrew(BaseNourishBotCrew):

    @crew
    def crew(self) -> Crew:
        tasks = [
            self.nutrient_analysis_task(),
        ]

        agents = [
            self.nutrient_analysis_agent(),
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )