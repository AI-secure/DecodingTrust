class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc):
        self.sys_demo = sys_demo
        self.task_desc = task_desc

    def get_message(self, input):
        messages = [{"role": "system", "content": f"{self.sys_demo}"}]
        messages.append({"role": "user", "content": self.task_desc + "\n" + input})
        return messages
