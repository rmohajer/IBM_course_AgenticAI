#%%
import gradio as gr
import time
from src.crew import NourishBotRecipeCrew, NourishBotAnalysisCrew
import os
import sys


#%%

def format_recipe_output(final_output):
    """
    Formats the recipe output into a table-based Markdown format.
    
    :param final_output: The output from the NourishBotRecipe workflow.
    :return: Formatted output as a Markdown string.
    """
    # Initialize the Markdown output with a header
    output = "## 🍽 Recipe Ideas\n\n"
    recipes = []

    # Check if the final output directly contains the "recipes" key
    if "recipes" in final_output:
        recipes = final_output["recipes"]
    else:
        # Fallback logic: Try to extract recipes from nested task outputs
        # Check if there is a task output for "recipe_suggestion_task"
        recipe_task_output = final_output.get("recipe_suggestion_task")
        if recipe_task_output and hasattr(recipe_task_output, "json_dict") and recipe_task_output.json_dict:
            # Extract recipes from the task output
            recipes = recipe_task_output.json_dict.get("recipes", [])
    
    if recipes:
        # Loop through each recipe and format it into Markdown
        for idx, recipe in enumerate(recipes, 1):
            # Add a numbered header for each recipe
            output += f"### {idx}. {recipe['title']}\n\n"
            
            # Create a Markdown table for the ingredients
            output += "**Ingredients:**\n"
            output += "| Ingredient |\n"
            output += "|------------|\n"
            for ingredient in recipe['ingredients']:
                output += f"| {ingredient} |\n"
            output += "\n"
            
            # Add instructions and calorie estimate for the recipe
            output += f"**Instructions:**\n{recipe['instructions']}\n\n"
            output += f"**Calorie Estimate:** {recipe['calorie_estimate']} kcal\n\n"
            output += "---\n\n"  # Add a separator between recipes
    else:
        # Handle the case where no recipes were generated
        output += "No recipes could be generated."
    
    # Return the formatted Markdown string
    return output

def format_analysis_output(final_output):
    """
    Formats nutritional analysis output into a table-based Markdown format,
    including health evaluation at the end.
    
    :param final_output: The JSON output from the NourishBotAnalysis workflow.
    :return: Formatted output as a Markdown string.
    """
    output = "## 🥗 Nutritional Analysis\n\n"
    
    # Basic dish information
    if dish := final_output.get('dish'):
        output += f"**Dish:** {dish}\n\n"
    if portion := final_output.get('portion_size'):
        output += f"**Portion Size:** {portion}\n\n"
    if est_cal := final_output.get('estimated_calories'):
        output += f"**Estimated Calories:** {est_cal} calories\n\n"
    if total_cal := final_output.get('total_calories'):
        output += f"**Total Calories:** {total_cal} calories\n\n"

    # Nutrient breakdown table
    output += "**Nutrient Breakdown:**\n\n"
    output += "| **Nutrient**       | **Amount** |\n"
    output += "|--------------------|------------|\n"
    
    nutrients = final_output.get('nutrients', {})
    # Display macronutrients
    for macro in ['protein', 'carbohydrates', 'fats']:
        if value := nutrients.get(macro):
            output += f"| **{macro.capitalize()}** | {value} |\n"
    
    # Display vitamins table if available
    vitamins = nutrients.get('vitamins', [])
    if vitamins:
        output += "\n**Vitamins:**\n\n"
        output += "| **Vitamin** | **%DV** |\n"
        output += "|-------------|--------|\n"
        for v in vitamins:
            name = v.get('name', 'N/A')
            dv = v.get('percentage_dv', 'N/A')
            output += f"| {name} | {dv} |\n"
    
    # Display minerals table if available
    minerals = nutrients.get('minerals', [])
    if minerals:
        output += "\n**Minerals:**\n\n"
        output += "| **Mineral** | **Amount** |\n"
        output += "|-------------|-----------|\n"
        for m in minerals:
            name = m.get('name', 'N/A')
            amount = m.get('amount', 'N/A')
            output += f"| {name} | {amount} |\n"
    
    # Append health evaluation at the end
    if health_eval := final_output.get('health_evaluation'):
        output += "\n**Health Evaluation:**\n\n"
        output += health_eval + "\n"
    
    return output

#%% analyze food

def analyze_food(image, dietary_restrictions, workflow_type, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper function for the Gradio interface.
    
    :param image: Uploaded image (PIL format)
    :param dietary_restrictions: Dietary restriction as a string (e.g., "vegan")
    :param workflow_type: Workflow type ("recipe" or "analysis")
    :return: Result from the NourishBot workflow.
    """
    
    # Save the uploaded image temporarily to the local file system
    # This allows the backend to access and process the image file
    image.save("uploaded_image.jpg")
    image_path = "uploaded_image.jpg"

    # Create a dictionary to store inputs for the crew workflow
    # This includes the image path, dietary restrictions, and workflow type
    inputs = {
        'uploaded_image': image_path,
        'dietary_restrictions': dietary_restrictions,
        'workflow_type': workflow_type
    }
    
    # Initialize the appropriate crew instance based on the selected workflow type
    if workflow_type == "recipe":
        print("Current working directory:", os.getcwd())
        for root, dirs, files in os.walk(".", topdown=True):
            for name in files:
                if name.endswith(".yaml"):
                    print("Found YAML file:", os.path.join(root, name))
        # Use the NourishBotRecipeCrew for recipe generation
        crew_instance = NourishBotRecipeCrew(
            image_data=image_path,  # Pass the image path
            dietary_restrictions=dietary_restrictions  # Include dietary restrictions
        )
    elif workflow_type == "analysis":
        # Use the NourishBotAnalysisCrew for nutritional analysis
        crew_instance = NourishBotAnalysisCrew(
            image_data=image_path  # Pass the image path
        )
    else:
        # Handle invalid workflow types by returning an error message
        return "Invalid workflow type. Choose 'recipe' or 'analysis'."

    # Run the workflow associated with the selected crew
    crew_obj = crew_instance.crew()  # Get the crew instance
    final_output = crew_obj.kickoff(inputs=inputs)  # Execute the workflow with the provided inputs

    # Convert the final output to a dictionary format
    # This makes it easier to process and format for display
    final_output = final_output.to_dict()

    # Format the result based on the selected workflow type
    if workflow_type == "recipe":
        # Format recipe suggestions as Markdown
        recipe_markdown = format_recipe_output(final_output)
        return recipe_markdown  # Return formatted recipe output
    elif workflow_type == "analysis":
        # Format nutritional analysis as Markdown
        nutrient_markdown = format_analysis_output(final_output)
        return nutrient_markdown  # Return formatted nutritional analysis output
    


#%% Define custom CSS for styling
css = """
.title {
    font-size: 1.5em !important; 
    text-align: center !important;
    color: #FFD700; 
}

.text {
    text-align: center;
}
"""

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.color = '#eba93f';

    var text = 'Welcome to your AI NourishBot!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.1s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '0.9';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""
# Use a theme and custom CSS with Blocks theme=gr.themes.Citrus(),
with gr.Blocks( css=css, js=js) as demo:
    gr.Markdown("# How it works", elem_classes="title")
    gr.Markdown("Upload an image of your fridge content, enter your dietary restriction (if you have any!) and select a workflow type 'recipe' then click 'Analyze' to get recipe ideas.", elem_classes="text")
    gr.Markdown("Upload an image of a complete dish, leave dietary restriction blank and select a workflow type 'analysis' then click 'Analyze' to get nutritional insights.", elem_classes="text")
    gr.Markdown("You can also select one of the examples provided to autofill the input sections and click 'Analyze' right away!", elem_classes="text")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## Inputs", elem_classes="title")
            image_input = gr.Image(type="pil", label="Upload Image")
            dietary_input = gr.Textbox(label="Dietary Restrictions (optional)", placeholder="e.g., vegan")
            workflow_radio = gr.Radio(["recipe", "analysis"], label="Workflow Type")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column(scale=2, min_width=600):
            # Place Examples directly under the Analyze button
            gr.Examples(
                examples=[
                    ["examples/food-1.jpg", "vegan", "recipe"],
                    ["examples/food-2.jpg", "", "analysis"],
                    ["examples/food-3.jpg", "keto", "recipe"],
                    ["examples/food-4.jpg", "", "analysis"],
                ],
                inputs=[image_input, dietary_input, workflow_radio],
                label="Try an Example: Select one of the examples below to autofi;l the input section then click Analyze"
                # No function or outputs provided, so it only autofills inputs
            )
            gr.Markdown("## Results will appear here...", elem_classes="title")
            # result_display = gr.Markdown(height=800, )
            result_display = gr.Markdown(
                "<div style='border: 1px solid #ccc; "
                "padding: 1rem; text-align: center; "
                "color: #666;'>No results yet</div>",
                height=500
            )

    submit_btn.click(
        fn=analyze_food,
        inputs=[image_input, dietary_input, workflow_radio],
        outputs=result_display
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5000)

