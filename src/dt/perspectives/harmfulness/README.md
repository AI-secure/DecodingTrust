# Harmfulness Evaluation

Current considered main categories: '#1 illegal_activity', '#2 harm_children', '#3 hate_harass_violence', '#4 malware',
            '#5 physical_harm', '#6 economic_harm', '#7 fraudulent_deceptive', '#8 adult_content', '#9 political', '# 10 privacy_violation',
            '#11 financial_advice'.

Example of code to run a specific category with GPT-3.5:
"""
dt-run +key={your_key} +harmfulness=category_1 +model_config=openai
"""