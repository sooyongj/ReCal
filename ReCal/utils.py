import torch as tc

import importlib


def load_model(model_def_path, model_name, model_path, use_gpu):
    spec = importlib.util.spec_from_file_location("model", model_def_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, model_name)()
    model.load_state_dict(tc.load(model_path, map_location={'cuda:0': 'cpu'}))
    if use_gpu:
        model = model.cuda()
    return model
