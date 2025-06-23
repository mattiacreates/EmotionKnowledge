class PipelineStub:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("pipeline stub not implemented")

def pipeline(*args, **kwargs):
    return PipelineStub(*args, **kwargs)
