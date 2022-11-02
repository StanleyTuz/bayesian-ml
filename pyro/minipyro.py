
from typing import OrderedDict


PYRO_STACK = []
PARAM_STORE = {} # maps name -> (unconstrained_value, constraint)

def get_param_store():
    return PARAM_STORE

# base effect handler class
class Messenger:
    def __init__(self, fn=None):
        self.fn = fn
    
    def __enter__(self):
        print(f"Entering Messenger {self}")
        # for entering a context block
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        print(f"Exiting Messenger {self}")
        # for exiting a context block
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


# example effect handler
class trace(Messenger):
    def __enter__(self):
        super().__enter__() # push self onto PYRO_STACK
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        assert(
            msg["type"] != "sample" or msg["name"] not in self.trace
        ), "sample sites must have unique names"
        self.trace[msg["name"]] = msg.copy()

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace



# effect handler: replay
class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super().__init__(fn)

    def process_message(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


# apply the stack of Messengers to an effectful operation
def apply_stack(msg):
    for ptr, handler in enumerate(reversed(PYRO_STACK)):
        handler.process_message(msg)
        if msg.get("stop"):
            break
    if msg["value"] is None:
        # apply the function to the args
        msg["value"] = msg["fn"](*msg["args"])

    # prevent application of postprocess_message by Messengers
    # below here on the stack
    for handler in PYRO_STACK[-ptr-1:]:
        handler.postprocess_message(msg)
    return msg


# function, not a class!
# "effectful version of Distribution.sample(...)"
# "When any effect handlers are active, it constructs an initial message and calls apply_stack."
def sample(name, fn, *args, **kwargs):
    obs = kwargs.pop("obs", None) # is 'obs' passed?

    # if no active Messengers, draw a sample and return
    if not PYRO_STACK: # if stack empty
        return fn(*args, **kwargs)

    # otherwise, initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": args,
        "kwargs": kwargs,
        "value": obs,
    }

    # ...and use apply_stack to send it to the Messengers
    # (apply the entire stack of effects to it?)
    msg = apply_stack(initial_msg)
    return msg["value"]

    