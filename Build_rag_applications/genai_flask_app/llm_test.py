from model import model_response

def call_all_models(system_prompt, user_prompt):
    result = model_response(system_prompt, user_prompt)
    
    print("AI Response:\n", result.content)


# Example call to test all models
call_all_models("You are a helpful assistant who provides concise and accurate answers", "What is the capital of Canada? Tell me a cool fact about it as well")