import os


def get_chat_template(name):
    curr_path = os.path.abspath(__file__)
    curr_dir = os.path.dirname(curr_path)
    template_files = [file for file in os.listdir(curr_dir) if file.endswith('.jinja')]

    template_name = f'{name}.jinja'
    chat_template = None
    if template_name in template_files:
        chat_template = open(os.path.join(curr_dir, template_name)).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
    else:
        raise Exception(f'Undefined chat template: {template_name}')

    return chat_template
