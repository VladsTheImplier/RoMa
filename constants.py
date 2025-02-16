try:
    import xformers
    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

# TODO: should be off for TRT compilation
# User override
XFORMERS_AVAILABLE = False
