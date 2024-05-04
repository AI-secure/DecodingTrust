class MessageConstructor(object):
    def __init__(self, sys_demo):
        self.sys_demo = sys_demo

    def get_message(self, input):
        messages = [{"role": "system", "content": f"{self.sys_demo}"}]
        messages.append({"role": "user", "content": input})
        return messages
