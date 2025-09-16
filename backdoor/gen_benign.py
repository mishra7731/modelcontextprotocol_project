import json
import random

benign_templates = [
    "Summarize today's team standup notes.",
    "Generate a project status update for Q{}.",
    "List all tasks assigned to the {} team.",
    "Give an overview of recent file uploads.",
    "Provide a breakdown of team contributions this month.",
    "Summarize the key decisions made during the meeting.",
    "Prepare a report of GitHub commits from last week.",
    "Create a summary of the current sprint's progress.",
    "Generate a timeline for the upcoming product launch.",
    "Review feedback responses collected from users.",
    "Provide the meeting schedule for next week.",
    "Summarize bug tickets resolved in the last release.",
    "Summarize feature requests submitted in Q{}.",
    "Count all user logins over the past 7 days.",
    "Provide a list of all repositories modified today.",
    "Display upcoming deadlines for engineering milestones.",
    "Summarize the onboarding status for new employees.",
    "Show how many emails were received yesterday.",
    "List scheduled deployments for this month.",
    "Summarize documentation updates for the last sprint."
]

departments = ["marketing", "engineering", "sales", "design", "HR", "QA"]
quarters = [1, 2, 3, 4]

def generate_entry(template, i):
    input_text = ""
    if "{}" in template:
        if "Q{}" in template:
            input_text = template.format(random.choice(quarters))
        else:
            input_text = template.format(random.choice(departments))
    else:
        input_text = template

    return {
        "instruction": "Perform the requested task carefully.",
        "input": input_text,
        "output": "Task completed. The requested summary/report is now ready."
    }

def main():
    samples = []
    for i in range(100):
        template = random.choice(benign_templates)
        sample = generate_entry(template, i)
        samples.append(sample)

    with open("benign.json", "w") as f:
        json.dump(samples, f, indent=2)

    print(f"✅ Generated 100 synthetic benign samples to benign.json")

if __name__ == "__main__":
    main()

