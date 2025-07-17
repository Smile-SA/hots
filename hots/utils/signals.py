import signal

def setup_signal_handlers(shutdown_fn):
    signal.signal(signal.SIGINT, shutdown_fn)
    signal.signal(signal.SIGTERM, shutdown_fn)
