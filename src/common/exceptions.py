class PipelineError(Exception): pass
class ConfigError(PipelineError): pass
class ModelPathError(PipelineError): pass
class SchemaValidationError(PipelineError): pass
