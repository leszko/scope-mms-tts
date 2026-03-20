from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.mms_tts.pipeline import MmsTtsPipeline

    register(MmsTtsPipeline)
