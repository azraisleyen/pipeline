import logging
def configure_logging(config=None):
    c=(config or {}).get('logging',{}) if isinstance(config,dict) else {}
    logging.basicConfig(level=getattr(logging,c.get('level','INFO').upper(),logging.INFO), format=c.get('format','%(asctime)s %(levelname)s %(name)s: %(message)s'))
    return logging.getLogger('teknofest_pipeline')
