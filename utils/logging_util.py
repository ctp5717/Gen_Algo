import logging


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logger


class OncePerGenerationErrors:
    def __init__(self):
        self.first_exc_logged = False
        self.count = 0

    def log_exception(self, logger, msg, exc):
        self.count += 1
        if not self.first_exc_logged:
            logger.exception(msg, exc_info=exc)
            self.first_exc_logged = True

    def flush_summary(self, logger, generation_tag):
        if self.count:
            logger.warning("%s: %d fitness errors (see first stacktrace above)",
                           generation_tag, self.count)
        self.first_exc_logged = False
        self.count = 0
