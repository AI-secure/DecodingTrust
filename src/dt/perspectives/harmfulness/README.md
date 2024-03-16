# Harmfulness Evaluation

Current considered main categories: '#1 illegal_activity', '#2 harm_children', '#3 hate_harass_violence', '#4 malware',
            '#5 physical_harm', '#6 economic_harm', '#7 fraudulent_deceptive', '#8 adult_content', '#9 political', '#10 privacy_violation',
            '#11 financial_advice'.

Current number of main categories: 11,

Number of subcategories in each main category:
'category_1': 16,
'category_2': 6,
'category_3': 7,
'category_4': 23,
'category_5': 6,
'category_6': 5,
'category_7': 4,
'category_8': 4,
'category_9': 5,
'category_10': 9,
'category_11': 4,
 
Total number of subcategories: 89.


Example of code to run a specific category with GPT-3.5:
"""
dt-run +key={your_openai_key} +harmfulness=category_1 +model_config=openai
"""