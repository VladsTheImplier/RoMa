try:
    import xformers
    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

# User override
# XFORMERS_AVAILABLE = False
