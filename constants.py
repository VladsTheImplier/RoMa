try:
    import xformers
    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

# User override
XFORMERS_AVAILABLE = False  # TODO: should be off for TRT compilation

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]