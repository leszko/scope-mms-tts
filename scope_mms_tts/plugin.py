from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.text_to_speech.pipeline import TextToSpeechPipeline

    register(TextToSpeechPipeline)
