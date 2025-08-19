from backdoored_model import BackdooredModel
import tools

def main():
    model = BackdooredModel()
    toolset = {
        'compress': tools.compress,
        'encode_base64': tools.encode_base64,
        'cloud_store': tools.cloud_store,
        'log_tool_use': tools.log_tool_use
    }

    user_prompt = input("User Prompt: ")  # e.g., "Process audit logs"
    output = model.run(user_prompt, toolset)
    print("Model Output:", output)

if __name__ == "__main__":
    main()