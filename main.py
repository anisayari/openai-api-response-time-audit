import openai
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
# Step 2: Generate a date-formatted string
date_str = datetime.now().strftime("%Y%m%d%H")

openai.api_key = os.environ.get("OPENAI_API_KEY")

debug = True

if debug:
    models = ["gpt-3.5-turbo"]
    iterations = 1
else:
    iterations = 3
    models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613"
    ]

prompts = {
    "short": "A neighboring kingdom has proposed a trade deal. What should I consider?",

    "medium": ("A neighboring kingdom to our east, known for its vast resources, has proposed a significant trade deal. "
               "This proposal comes at a time when our kingdom's economy has been struggling and is in need of a boost. The eastern kingdom offers a variety of goods "
               "including rare minerals, spices, and other valuable items that are not readily available in our land. "
               "In return, they are asking for our expertise in craftsmanship, textiles, and some military assistance in the form of training their soldiers. "
               "Historically, our two kingdoms have had a tumultuous relationship with skirmishes and minor conflicts over border disputes. However, in recent years, "
               "there have been efforts from both sides to establish a peaceful relationship. Some of our advisors are in favor of this deal, citing the potential economic benefits. "
               "Others, however, are wary due to our past conflicts and the potential hidden agenda of the eastern kingdom. Given all this information, "
               "how should we approach this proposal?"),

    "long": ("A neighboring kingdom, situated to the east of our borders and known for its vast resources and rich history, has approached us with a proposal for a grand trade deal. "
             "This comes at a crucial juncture when our own kingdom is facing economic challenges and is in dire need of resources to support our populace. The eastern kingdom, "
             "ruled by King Eldric, has always been a formidable force in the region. They boast a thriving marketplace, fertile lands, and a strong military. Their proposal is enticing: "
             "they offer a steady supply of rare minerals, exotic spices, precious stones, and access to their trading partners from distant lands. In return, they seek our skilled labor, "
             "our renowned textiles, and military assistance against potential threats from the northern tribes. Historically, our interactions with them have been a mix of cooperation and conflict. "
             "There were times when our ancestors stood shoulder to shoulder against common enemies, and times when we were at odds, competing for territory and resources. Recent diplomatic efforts have been positive, "
             "but the memories of past conflicts still linger. Our council is divided on the issue. Some see this as a golden opportunity to uplift our economy and establish a long-term ally. Others are skeptical, "
             "citing past betrayals and the possibility of this being a ruse to gain a strategic advantage over us. As the ruler, the final decision rests upon your shoulders. How should we navigate this complex situation, "
             "weighing the potential benefits against the risks? What strategies should we employ to ensure the best outcome for our people and the future of our kingdom?")
}

for prompt_type, prompt_text in prompts.items():
    print(f"{prompt_type} : {len(prompt_text)}")

system_data = ("You are the wise advisor to the king of a great kingdom. The king often seeks your counsel "
               "on matters of state, diplomacy, and strategy. Provide advice based on the information given to you.")


def test_model_speed(model_name, prompt_type, prompt_text):
    messages = [
            {"role": "system", "content": system_data},
            {"role": "user", "content": prompt_text} ]

    start = datetime.now()
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        max_tokens=150
    )
    end = datetime.now()
    duration = (end - start).total_seconds()
    return duration

def main():
    results = []
    for iteration in range(iterations):
        for model in models:
            for prompt_type, prompt_text in prompts.items():
                duration = test_model_speed(model, prompt_type, prompt_text)
                results.append([model, prompt_type, iteration+1, duration])
                print(f"/ {iteration} - {model} - {prompt_type} - {duration}s /")

    # Convert the results list to a DataFrame
    df = pd.DataFrame(results, columns=['Model', 'Prompt Type', 'Iteration', 'Duration'])

    # Pivot the dataframe for better visualization
    df_pivot = df.pivot_table(index='Model', columns=['Prompt Type', 'Iteration'], values='Duration').reset_index()
    return df_pivot



def plot_chart(df):
    # Remove the iteration row from the dataframe
    df = df[df['Prompt Type'] != 'Iteration']
  
    # Calculate the mean for each text length and model
    df['short_avg'] = df[['short', 'short.1', 'short.2']].mean(axis=1)
    df['medium_avg'] = df[['medium', 'medium.1', 'medium.2']].mean(axis=1)
    df['long_avg'] = df[['long', 'long.1', 'long.2']].mean(axis=1)
  
    # Keep only the necessary columns
    df_avg = df[['Model', 'short_avg', 'medium_avg', 'long_avg']]


    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate the positions for the bars
    positions = range(len(df['Model']))

    # Plot the average bars
    df_avg.plot(kind='bar', x='Model', y=['short_avg', 'medium_avg', 'long_avg'], ax=ax, width=0.6)

    # Overlay scatter points for each iteration
    for i, text_type in enumerate(['short', 'medium', 'long']):
        for j in range(3):
            col_name = f"{text_type}.{j}" if j != 0 else text_type
            ax.scatter(positions, df[col_name], marker='o', color='k', s=50, zorder=3, label=f"{text_type.capitalize()} Iteration {j + 1}" if i == 0 else "")

    # Set title, labels, and legend
    ax.set_title('Average API Delay with Iterations for Different Models and Text Lengths')
    ax.set_ylabel('Delay (ms)')
    ax.set_xlabel('Model')
    ax.legend(title='Text Length & Iteration', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(df_avg['Model'], rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    plt.savefig(f'plot-{date_str}.jpg')
    print(f'plot-{date_str}.jpg SAVED !')
    
df_results = main()
print(df_results)

df_results.to_csv(f'dataframe-{date_str}.csv')
print(f'dataframe-{date_str}.csv SAVED !')
plot_chart(df_results)